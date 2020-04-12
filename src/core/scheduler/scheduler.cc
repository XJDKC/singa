/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "singa/core/scheduler.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <sstream>
#include <thread>
#include <unordered_set>

#include "singa/core/device.h"
#include "singa/utils/safe_queue.h"

namespace singa {

void Node::AddInEdge(Edge *in_edge) { in_edges_.push_back(in_edge); }

void Node::AddOutEdge(Edge *out_edge) { out_edges_.push_back(out_edge); }

void Edge::SetBlock(Block *blk) { blk_ = blk; }

void Edge::SetSrcNode(Node *src_node) { src_node_ = src_node; }

void Edge::SetDstNode(Node *dst_node) { dst_node_ = dst_node; }

OpRec::OpRec() : time_(0) {
  cudaEventCreateWithFlags(&start_, cudaEventBlockingSync);
  cudaEventCreate(&end_, cudaEventBlockingSync);
}

Graph::Graph(Device *device)
    : device_(device), thread_(&Graph::FreeLoop, this) {
  cudaStreamCreateWithFlags(&swap_, cudaStreamNonBlocking);
  autoswap_ = device->id() != -1;
  // autoswap_ = false;
}

Graph::~Graph() {
  thread_.join();
  Reset();
}

Node *Graph::node(const size_t idx) const {
  CHECK_LT(idx, nodes_.size());
  return nodes_[idx];
}

Edge *Graph::edge(const size_t idx) const {
  CHECK_LT(idx, edges_.size());
  return edges_[idx];
}

BlkInfo *Graph::block(Block *blk) const {
  auto it = blocks_.find(blk);
  CHECK(it != blocks_.end());
  return it->second;
}

Block *Graph::write_block(const size_t idx) const {
  CHECK_LT(idx, write_blocks_.size());
  return write_blocks_[idx];
}

Node *Graph::begin_node(const size_t idx) const {
  CHECK_LT(idx, begin_nodes_.size());
  return begin_nodes_[idx];
}

const NodeVec &Graph::next_nodes(const size_t idx) const {
  CHECK_LT(idx, next_nodes_.size());
  return next_nodes_[idx];
}

const BlockVec &Graph::free_blocks(const size_t idx) const {
  CHECK_LT(idx, free_blocks_.size());
  return free_blocks_[idx];
}

void Graph::Reset() {
  for (auto it : nodes_) {
    delete it;
  }
  nodes_.clear();

  for (auto it : edges_) {
    delete it;
  }
  edges_.clear();

  for (auto it : blocks_) {
    delete it.second;
  }
  blocks_.clear();

  for (auto it : host_blks_) {
    delete it;
  }
  host_blks_.clear();

  write_blocks_.clear();

  ResetPlan();

  start_up_ = true;

  dirty_ = false;
}

void Graph::Debug() {
  if (dirty_) Analysis();

  int w = 0;
  size_t max_in_num = 0, max_out_num = 0, max_next_num = 0, max_free_num = 0;
  for (auto &it : nodes_) {
    max_in_num = std::max(max_in_num, it->in_edges_.size());
    max_out_num = std::max(max_out_num, it->out_edges_.size());
  }

  for (auto &it : next_nodes_) {
    max_next_num = std::max(max_next_num, it.size());
  }

  for (auto &it : free_blocks_) {
    max_free_num = std::max(max_free_num, it.size());
  }

  for (int i = std::max(nodes_.size(), blocks_.size()); i > 0; i /= 10, ++w)
    ;

  std::stringstream ss;
  ss << "begin nodes:[";
  for (size_t i = 0; i < begin_nodes_.size(); ++i) {
    ss << std::setw(w) << begin_nodes_[i]->id_ << " ";
  }
  ss << "]" << std::endl;

  size_t size = 0;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    ss << "OP[" << std::setw(w) << i;
    auto node = nodes_[i];

    ss << "] Inputs:[";
    size = node->in_edges_.size();
    for (size_t j = 0; j < max_in_num; ++j) {
      if (j < size)
        ss << std::setw(w) << blocks_[node->in_edges_[j]->blk_]->id_ << " ";
      else
        ss << std::setw(w + 1) << " ";
    }

    ss << "] Outputs:[";
    size = node->out_edges_.size();
    for (size_t j = 0; j < max_out_num; ++j) {
      if (j < size)
        ss << std::setw(w) << blocks_[node->out_edges_[j]->blk_]->id_ << " ";
      else
        ss << std::setw(w + 1) << " ";
    }

    ss << "] Next nodes:[";
    size = next_nodes_[i].size();
    for (size_t j = 0; j < max_next_num; ++j) {
      if (j < size)
        ss << std::setw(w) << next_nodes_[i][j]->id_ << " ";
      else
        ss << std::setw(w + 1) << " ";
    }

    ss << "] Free blocks:[";
    size = free_blocks_[i].size();
    for (size_t j = 0; j < max_free_num; ++j) {
      if (j < size)
        ss << std::setw(w) << blocks_[free_blocks_[i][j]]->id_ << " ";
      else
        ss << std::setw(w + 1) << " ";
    }
    ss << "]" << std::endl;
  }

  size_t max_used_num = 0;
  std::vector<BlkInfo *> blkInfos;
  blkInfos.resize(blocks_.size());

  for (auto it : blocks_) {
    blkInfos[it.second->id_] = it.second;
    max_used_num = std::max(max_used_num, it.second->used_nodes_.size());
  }

  for (auto it : blkInfos) {
    auto blkInfo = it;
    ss << "Block[" << std::setw(w) << blkInfo->id_ << "] addr[" << std::setw(w)
       << blkInfo->blk_ << "] size[" << std::setw(9) << it->blk_->size()
       << "] graph_ref[" << std::setw(w) << blkInfo->graph_ref_
       << "] ref_count[" << std::setw(w) << blkInfo->blk_->ref_count() << "] ";
    switch (blkInfo->type_) {
      case BlockType::kInput:
        ss << "type[input] ";
        break;
      case BlockType::kParam:
        ss << "type[param] ";
        break;
      case BlockType::kInter:
        ss << "type[inter] ";
        break;
      case BlockType::kEnd:
        ss << "type[_end_] ";
        break;
      default:
        break;
    }
    int id = -1;
    if (blkInfo->write_edge_) {
      id = blkInfo->write_edge_->src_node_->id_;
    }
    ss << " write_node[" << std::setw(w) << id << "]";

    size = blkInfo->used_nodes_.size();
    ss << " used_nodes[";
    for (size_t i = 0; i < max_used_num; ++i) {
      if (i < size)
        ss << std::setw(w) << blkInfo->used_nodes_[i]->id_ << " ";
      else
        ss << std::setw(w + 1) << " ";
    }
    ss << "]" << std::endl;
  }

  printf("%s", ss.str().c_str());
}

void Graph::RunGraph() {
  if (dirty_) Analysis();

  Context *ctx = device_->context(0);

  SafeQueue<Node *> node_queue;

  // activate nodes
  for (auto it : begin_nodes_) {
    node_queue.Push(it);
  }

  // run graph
  while (node_queue.Size()) {
    // step 1: pop the first element, get the node corresponding to the index
    Node *curNode = nullptr;
    node_queue.Pop(curNode);
    int curIndex = curNode->id_;

    if (autoswap_) {
      // step 2: swap in blocks if autoswap is enabled
      for (auto &it : swap_in_[curIndex]) {
        // printf("swap in Block[%d] Op[%d] ", blocks_[it->device_blk_]->id_,
        // curIndex);
        SwapBlock(it, true);
      }

      // step 3: wait for the blocks used by curNode to swap in
      for (auto &it : swap_wait_[curIndex]) {
        // printf("wait Block[%d] OP[%d] \n", blocks_[it->device_blk_]->id_,
        // curIndex);
        CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, it->in_rec_.end_, 0));
      }

      if (start_up_) {
        // for getting the elasped time of curNode
        CUDA_CHECK(cudaEventRecord(node_recs_[curIndex].start_, ctx->stream));
      }
    }

    // step 4: execute the operation
    device_->DoExec(std::move(curNode->op_), 0);

    if (autoswap_) {
      if (start_up_ || swap_out_[curIndex].size()) {
        // record a event if some blocks have to swap out
        CUDA_CHECK(cudaEventRecord(node_recs_[curIndex].end_, ctx->stream));
        if (swap_out_[curIndex].size()) {
          // printf("wait OP[%d] ", curIndex);
        }
      }

      // step 5: swap out blocks if autoswap is enbaled
      if (autoswap_) {
        for (auto &it : swap_out_[curIndex]) {
          // printf("swap out Block[%d] Op[%d] ", blocks_[it->device_blk_]->id_,
          // curIndex);
          SwapBlock(it, false);
        }
      }
    }

    // step 6: release some blocks' data that won't be used later
    for (auto it : free_blocks_[curIndex]) {
      it->free_data();
    }

    // step 7: activate the following nodes
    for (auto it : next_nodes_[curIndex]) {
      node_queue.Push(it);
    }
  }

  if (start_up_) {
    start_up_ = false;
    if (autoswap_) {
      RecordTime();
      AutoSwap();
    }
  }
}

void Graph::RunInSerial() {
  if (dirty_) Analysis();

  for (size_t i = 0; i < nodes_.size(); ++i) {
    Node *curNode = nodes_[i];

    // step 1: execute the operation
    device_->DoExec(std::move(curNode->op_), 0);

    // step 2: release some blocks' data that won't be used later
    for (auto it : free_blocks_[i]) {
      it->free_data();
    }
  }
}

void Graph::AddOperation(OpFunc &&op, const BlockVec &read_blocks,
                         const BlockVec &write_blocks) {
  dirty_ = true;
  start_up_ = true;

  node_recs_.resize(nodes_.size() + 1);

  // if the size of both read_blocks and write_blocks is zero,
  // this operation is used for synchronization
  if (read_blocks.size() == 0 && write_blocks.size() == 0) {
    AddSyncOp(std::move(op));
    return;
  }

  // create new node
  Node *node = new Node(nodes_.size(), std::move(op));

  // create a set to determine if there is a loop
  std::unordered_set<Block *> circle;

  // create edges for read_blocks
  for (size_t i = 0; i < read_blocks.size(); ++i) {
    Block *blk = read_blocks[i];
    Node *src_node = nullptr;
    BlkInfo *blkInfo = nullptr;

    auto it = blocks_.find(blk);
    if (it == blocks_.end()) {
      blkInfo = new BlkInfo(blocks_.size(), blk, BlockType::kInput);
      blocks_[blk] = blkInfo;
    } else {
      blkInfo = it->second;
      if (blkInfo->type_ == BlockType::kEnd) {
        blkInfo->type_ = BlockType::kInter;
      }

      Edge *write_edge = blkInfo->write_edge_;
      if (write_edge) {
        if (!write_edge->dst_node_) {
          // change the dst node of the write_edge
          blkInfo->used_nodes_.push_back(node);
          write_edge->dst_node_ = node;
          node->AddInEdge(write_edge);
          blkInfo->graph_ref_ += 1;
          circle.insert(blk);
          continue;
        } else {
          src_node = write_edge->src_node_;
        }
      }
    }

    Edge *edge = new Edge(edges_.size(), blk, src_node, node);
    blkInfo->used_nodes_.push_back(node);
    blkInfo->graph_ref_ += 1;
    if (src_node) {
      src_node->AddOutEdge(edge);
    }

    circle.insert(blk);
    node->AddInEdge(edge);
    edges_.push_back(edge);
  }

  // update last node for write_blocks
  for (size_t i = 0; i < write_blocks.size(); ++i) {
    Block *blk = write_blocks[i];
    BlkInfo *blkInfo = nullptr;

    auto it = blocks_.find(blk);
    if (it == blocks_.end()) {
      blkInfo = new BlkInfo(blocks_.size(), blk, BlockType::kEnd);
      blocks_[blk] = blkInfo;
    } else {
      blkInfo = it->second;
      if (blkInfo->type_ == BlockType::kInput) {
        blkInfo->type_ = BlockType::kParam;
      }
    }

    if (circle.find(blk) == circle.end()) {
      blkInfo->used_nodes_.push_back(node);
    }

    Edge *edge = new Edge(edges_.size(), blk, node, nullptr);
    blkInfo->write_edge_ = edge;
    blkInfo->graph_ref_ += 1;

    node->AddOutEdge(edge);
    edges_.push_back(edge);
  }

  // for sync op
  write_blocks_ = write_blocks;

  // add node into nodes
  nodes_.push_back(node);
}

void Graph::Analysis() {
  ResetPlan();

  // init node ref
  std::vector<int> node_ref_;
  node_ref_.resize(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i) {
    node_ref_[i] = nodes_[i]->in_edges_.size();
  }

  // find all input edges and decrease ref count of nodes
  for (size_t i = 0; i < edges_.size(); ++i) {
    Node *src_node = edges_[i]->src_node_;
    if (!src_node) {
      Node *node = edges_[i]->dst_node_;
      int nodeId = node->id_;
      node_ref_[nodeId] -= 1;
    }
  }

  // activate nodes
  SafeQueue<Node *> node_queue;
  for (size_t i = 0; i < node_ref_.size(); ++i) {
    if (node_ref_[i] == 0) {
      begin_nodes_.push_back(nodes_[i]);
      node_queue.Push(nodes_[i]);
    }
  }

  // run graph
  int idx = 0;
  std::vector<int> order;
  std::vector<int> id2order(nodes_.size(), -1);
  while (node_queue.Size()) {
    // step 1: pop the first element, get the node corresponding to the index
    Node *curNode = nullptr;
    node_queue.Pop(curNode);
    int curIndex = curNode->id_;
    order.push_back(curIndex);
    id2order[curIndex] = idx++;

    // step 2: release some blocks' data that won't be used later
    free_blocks_[curIndex].clear();
    for (size_t i = 0; i < curNode->in_edges_.size(); ++i) {
      Edge *edge = curNode->in_edges_[i];
      Block *blk = edge->blk_;
      BlkInfo *blkInfo = blocks_[blk];

      // if curnode is the last node accessing the block
      if (blkInfo->used_nodes_.back() == curNode) {
        BlockType type = blkInfo->type_;
        // if the block belongs to a inter tensor
        // and isn't refered on the Python Side
        if ((type == BlockType::kInter) &&
            blkInfo->graph_ref_ >= blk->ref_count()) {
          free_blocks_[curIndex].push_back(blk);
        }
      }
    }
    for (size_t i = 0; i < curNode->out_edges_.size(); ++i) {
      Edge *edge = curNode->out_edges_[i];
      Block *blk = edge->blk_;
      BlkInfo *blkInfo = blocks_[blk];

      // if curnode is the last node accessing the block
      if (blkInfo->used_nodes_.back() == curNode) {
        BlockType type = blkInfo->type_;
        // if the block belongs to a inter tensor
        // and isn't refered on the Python Side
        if ((type == BlockType::kEnd) &&
            blkInfo->graph_ref_ >= blk->ref_count()) {
          free_blocks_[curIndex].push_back(blk);
        }
      }
    }

    // step 3: decrease ref count of nodes and activate nodes
    next_nodes_[curIndex].clear();
    for (size_t i = 0; i < curNode->out_edges_.size(); ++i) {
      Edge *edge = curNode->out_edges_[i];
      Node *nextNode = edge->dst_node_;

      if (nextNode) {
        int nodeId = nextNode->id_;
        node_ref_[nodeId] -= 1;
        if (node_ref_[nodeId] <= 0) {
          node_queue.Push(nextNode);
          next_nodes_[curIndex].push_back(nextNode);
        }
      }
    }
  }

  // find candidate blocks for swapping
  host_blks_.resize(blocks_.size(), nullptr);
  for (auto &it : blocks_) {
    auto blk = it.first;
    auto blkInfo = it.second;
    auto type = blkInfo->type_;
    if (blk->size() >= threshold_ && blkInfo->used_nodes_.size() > 1 &&
        blkInfo->graph_ref_ >= blk->ref_count() &&
        (type == BlockType::kInter || type == BlockType::kEnd)) {
      auto &used_nodes = blkInfo->used_nodes_;
      for (size_t i = 1; i < used_nodes.size(); ++i) {
        int absense =
            id2order[used_nodes[i]->id_] - id2order[used_nodes[i - 1]->id_];

        if (absense <= nodes_.size() * 0.775) continue;

        // add candidate swap info
        int swap_out = used_nodes[i - 1]->id_;
        int swap_in = used_nodes[i]->id_;
        int next = used_nodes[i]->id_;

        int num = 150;
        if (id2order[swap_out] + 1 <= id2order[swap_in] - num) {
          swap_in = order[id2order[swap_in] - num];
        } else {
          swap_in = order[id2order[swap_out] + 1];
        }

        // printf("Swap Block[%d] size[%8ld] absense[%d] swap_out[%d]
        // swap_in[%d] next[%d]\n",
        //        blkInfo->id_, blk->size(), absense, swap_out, swap_in, next);

        Block *host_blk = host_blks_[blkInfo->id_];

        if (nullptr == host_blk) {
          host_blk = defaultDevice->NewBlock(blk->size());
          host_blk->mutable_data();
          host_blks_[blkInfo->id_] = host_blk;
        }

        SwapInfo *swap_info =
            new SwapInfo(next, swap_in, swap_out, host_blk, blk);
        swap_infos_.push_back(swap_info);

        swap_in_[swap_in].push_back(swap_info);
        swap_out_[swap_out].push_back(swap_info);
        swap_wait_[next].push_back(swap_info);
      }
    }
  }

  dirty_ = false;

  Debug();
}

void Graph::AutoSwap() {}

void Graph::ResetPlan() {
  begin_nodes_.clear();

  next_nodes_.clear();
  next_nodes_.resize(nodes_.size());

  free_blocks_.clear();
  free_blocks_.resize(nodes_.size());

  swap_in_.clear();
  swap_in_.resize(nodes_.size());

  swap_out_.clear();
  swap_out_.resize(nodes_.size());

  swap_wait_.clear();
  swap_wait_.resize(nodes_.size());

  for (size_t i = 0; i < swap_infos_.size(); ++i) {
    delete swap_infos_[i];
  }
  swap_infos_.clear();
}

void Graph::FreeLoop() {
  SwapInfo *swap_info = nullptr;
  for (;;) {
    free_queue_.Pop(swap_info);
    if (swap_info == nullptr) {
      break;
    } else {
      std::lock_guard<std::mutex> lck(swap_info->mtx_);
      if (!swap_info->on_device_) {
        swap_info->device_blk_->free_data();
        // printf("free block[%d] ", blocks_[swap_info->device_blk_]->id_);
      }
    }
  }
}

void Graph::RecordTime() {
  size_t size = node_recs_.size();

  if (size > 0) {
    CUDA_CHECK(cudaEventSynchronize(node_recs_[size - 1].end_));
  }

  float total_time = 0;
  for (size_t i = 0; i < size; ++i) {
    auto &rec = node_recs_[i];
    CUDA_CHECK(cudaEventElapsedTime(&rec.time_, rec.start_, rec.end_));
    // printf("OP[%ld] elapsedTime[%f]\n", i, rec.time_);
    total_time += rec.time_;
  }
  // printf("total_time[%f]\n", total_time);

  float total_in_time = 0;
  float total_out_time = 0;
  for (size_t i = 0; i < swap_infos_.size(); ++i) {
    auto &in_rec = swap_infos_[i]->in_rec_;
    auto &out_rec = swap_infos_[i]->out_rec_;

    CUDA_CHECK(cudaEventElapsedTime(&in_rec.time_, in_rec.start_, in_rec.end_));
    CUDA_CHECK(
        cudaEventElapsedTime(&out_rec.time_, out_rec.start_, out_rec.end_));

    // int id = blocks_[swap_infos_[i]->device_blk_]->id_;
    // printf("SwapIn: Block[%d] Time[%f]\n", id, in_rec.time_);
    // printf("SwapOut: Block[%d] Time[%f]\n", id, out_rec.time_);

    total_in_time += in_rec.time_;
    total_out_time += out_rec.time_;
  }

  // printf("total_in_time[%f] total_out_time[%f]\n", total_in_time,
  //        total_out_time);
}

void Graph::ReserveMem(size_t size) {
  size_t allocated_mem = device_->GetAllocatedMem();
  size_t reserve_mem = std::max(0ul, size - allocated_mem);
  Block *tmp = device_->NewBlock(reserve_mem);
  tmp->mutable_data();
  device_->FreeBlock(tmp);
}

void Graph::SwapBlock(SwapInfo *swap_info, bool direct) {
  Block *host_blk = swap_info->host_blk_;
  Block *device_blk = swap_info->device_blk_;

  // swap in the block if direct is true
  if (direct) {
    void *dst = nullptr;
    const void *src = host_blk->data();
    {
      std::lock_guard<std::mutex> lck(swap_info->mtx_);
      swap_info->on_device_ = true;
      dst = device_blk->mutable_data();
    }

    if (start_up_) {  // to record the swapping time
      CUDA_CHECK(cudaEventRecord(swap_info->in_rec_.start_, swap_));
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src, host_blk->size(),
                               cudaMemcpyHostToDevice, swap_));
    CUDA_CHECK(cudaEventRecord(swap_info->in_rec_.end_, swap_));
  } else {
    swap_info->on_device_ = false;
    const void *src = device_blk->data();
    void *dst = host_blk->mutable_data();

    // wait the operation to complete before swapping the block out
    CUDA_CHECK(
        cudaStreamWaitEvent(swap_, node_recs_[swap_info->swap_out_].end_, 0));

    // swap the block out of the device
    if (start_up_) {
      CUDA_CHECK(cudaEventRecord(swap_info->out_rec_.start_, swap_));
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src, device_blk->size(),
                               cudaMemcpyDeviceToHost, swap_));
    if (start_up_) {
      CUDA_CHECK(cudaEventRecord(swap_info->out_rec_.end_, swap_));
    }

    // free data if the swapping is complete
    CBData *cb_data = new CBData(this, swap_info);
    cudaStreamAddCallback(swap_, Graph::Callback, (void *)(cb_data), 0);
  }
}

void Graph::AddSyncOp(function<void(Context *)> &&op) {
  // create new node
  Node *node = new Node(nodes_.size(), std::move(op));

  for (size_t i = 0; i < write_blocks_.size(); ++i) {
    Block *blk = write_blocks_[i];
    BlkInfo *blkInfo = blocks_[blk];
    Edge *edge = nullptr;

    if (blkInfo->type_ == BlockType::kEnd) {
      blkInfo->type_ = BlockType::kInter;
    }

    Edge *write_edge = blkInfo->write_edge_;
    if (!write_edge->dst_node_) {
      // change the dst node of the write_edge
      write_edge->dst_node_ = node;
      edge = write_edge;
    } else {
      Node *src_node = write_edge->src_node_;
      edge = new Edge(edges_.size(), blk, src_node, node);
      src_node->AddOutEdge(edge);
      edges_.push_back(edge);
    }

    node->AddInEdge(edge);

    // fake edges, no need to add the graph ref
    edge = new Edge(edges_.size(), blk, node, nullptr);
    blkInfo->write_edge_ = edge;

    node->AddOutEdge(edge);
    edges_.push_back(edge);
  }

  // add node into nodes
  nodes_.push_back(node);
}

void CUDART_CB Graph::Callback(cudaStream_t stream, cudaError_t status,
                               void *data) {
  CBData *cbData = (CBData *)data;
  cbData->graph_->free_queue_.Push(cbData->swap_info_);
  delete cbData;
}

}  // namespace singa
