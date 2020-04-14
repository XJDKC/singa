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
#include <queue>
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
  CUDA_CHECK(cudaEventCreate(&start_));
  CUDA_CHECK(cudaEventCreate(&end_));
}

OpRec::~OpRec() {
  // cudaEventDestroy(start_);
  // cudaEventDestroy(end_);
}

Graph::Graph(Device *device)
    : device_(device), thread_(&Graph::ThreadLoop, this) {
  CUDA_CHECK(cudaStreamCreateWithFlags(&in_stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&out_stream_, cudaStreamNonBlocking));
  autoswap_ = device->id() != -1;
  // autoswap_ = false;
  thread_.detach();
}

Graph::~Graph() {
  CUDA_CHECK(cudaStreamDestroy(in_stream_));
  CUDA_CHECK(cudaStreamDestroy(out_stream_));
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
    CUDA_CHECK(cudaFreeHost(it->mutable_data()));
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
      // wait for the blocks which memory are freed to swap out
      for (auto &it : swap_free_[curIndex]) {
        it->device_blk_->free_data();
        CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, it->out_rec_.end_, 0));
        CUDA_CHECK(cudaStreamWaitEvent(in_stream_, it->out_rec_.end_, 0));
        // printf("ctx stream wait Swap Out End Event block[%d]\n",
        // blocks_[it->device_blk_]->id_); printf("in stream wait Swap Out End
        // Event block[%d]\n", blocks_[it->device_blk_]->id_); printf("free
        // block[%d] Op[%d]\n", blocks_[it->device_blk_]->id_, curIndex);
      }

      // step 2: swap in blocks if autoswap is enabled
      for (auto &it : swap_in_[curIndex]) {
        SwapBlock(it, true);
        // printf("swap in Block[%d] Op[%d] \n", blocks_[it->device_blk_]->id_,
        // curIndex);
      }

      // step 3: wait for the blocks used by curNode to swap in
      for (auto &it : swap_wait_[curIndex]) {
        CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, it->out_rec_.end_, 0));
        CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, it->in_rec_.end_, 0));
        // printf("ctx stream wait Swap Out Block[%d] OP[%d] \n",
        // blocks_[it->device_blk_]->id_, curIndex); printf("ctx stream wait
        // Swap In Block[%d] OP[%d] \n", blocks_[it->device_blk_]->id_,
        // curIndex);
      }

      if (start_up_) {
        // for getting the elasped time of curNode
        CUDA_CHECK(cudaEventRecord(node_recs_[curIndex].start_, ctx->stream));
      }
    }

    // step 4: execute the operation
    // printf("Exec OP[%d]\n", curIndex);
    device_->DoExec(std::move(curNode->op_), 0);

    if (autoswap_) {
      if (start_up_ || swap_out_[curIndex].size()) {
        // record a event if some blocks have to swap out
        // printf("Record End Event Op[%d]\n", curIndex);
        CUDA_CHECK(cudaEventRecord(node_recs_[curIndex].end_, ctx->stream));
      }

      // step 5: swap out blocks if autoswap is enbaled
      for (auto &it : swap_out_[curIndex]) {
        SwapBlock(it, false);
        // printf("swap out Block[%d] Op[%d] \n", blocks_[it->device_blk_]->id_,
        // curIndex);
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

  device_->Sync();
  CUDA_CHECK(cudaStreamSynchronize(in_stream_));
  CUDA_CHECK(cudaStreamSynchronize(out_stream_));
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
  ids_.clear();
  id2order_.clear();
  id2order_.resize(nodes_.size());
  while (node_queue.Size()) {
    // step 1: pop the first element, get the node corresponding to the index
    Node *curNode = nullptr;
    node_queue.Pop(curNode);
    int curIndex = curNode->id_;
    ids_.push_back(curIndex);
    id2order_[curIndex] = idx;
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

  if (autoswap_) {
    // init auto swap to get elapsed time of ops and swapping
    for (auto &it : blocks_) {
      auto blk = it.first;
      auto blkInfo = it.second;
      auto type = blkInfo->type_;
      auto &used_nodes = blkInfo->used_nodes_;
      if (blk->size() >= threshold_ && used_nodes.size() > 1 &&
          blkInfo->graph_ref_ >= blk->ref_count() &&
          (type == BlockType::kInter || type == BlockType::kEnd)) {
        size_t idx = 0;
        int max_absense = 0;
        for (size_t i = 1; i < used_nodes.size(); ++i) {
          int from = id2order[used_nodes[i - 1]->id_];
          int to = id2order[used_nodes[i]->id_];
          int absense = to - from;
          if (absense > max_absense) {
            max_absense = absense;
            idx = i - 1;
          }
        }

        if (max_absense <= 2) continue;

        // add candidate swap info
        int swap_out = used_nodes[idx]->id_;
        int swap_free = order[id2order[swap_out] + 1];
        int swap_in = used_nodes[idx + 1]->id_;
        int swap_next = used_nodes[idx + 1]->id_;

        printf("swap Block[%d] out[%d] free[%d] in[%d] next[%d]\n",
               blkInfo->id_, swap_out, swap_free, swap_in, swap_next);

        auto host_blk = host_blks_[blkInfo->id_];
        auto swap_info = new SwapInfo(swap_out, swap_free, swap_in, swap_next,
                                      host_blk, blk);
        swap_infos_.push_back(swap_info);

        swap_out_[swap_out].push_back(swap_info);
        swap_free_[swap_free].push_back(swap_info);
        swap_in_[swap_in].push_back(swap_info);
        swap_wait_[swap_next].push_back(swap_info);
      }
    }
  }

  dirty_ = false;

  // Debug();
}

void Graph::AutoSwap() {
  std::vector<int> ids;
  std::vector<int> id2order;

  id2order.resize(nodes_.size());
  ids.resize(begin_nodes_.size());
  for (size_t i = 0; i < begin_nodes_.size(); ++i) {
    ids[i] = begin_nodes_[i]->id_;
  }

  std::vector<size_t> chart;
  std::vector<bool> allocated(blocks_.size(), false);
  size_t peak_mem = device_->GetAllocatedMem();
  size_t current_mem = peak_mem;
  size_t idx = -1;
  for (size_t i = 0; i < ids.size(); ++i) {
    int curIndex = ids[i];
    Node *curNode = nodes_[curIndex];

    id2order[curIndex] = i;
    CHECK_EQ(ids[i], ids_[i]);
    CHECK_EQ(id2order[ids[i]], id2order_[ids_[i]]);
    for (auto &it : next_nodes_[curIndex]) {
      ids.push_back(it->id_);
    }

    for (auto &it : curNode->in_edges_) {
      Block *blk = it->blk_;
      BlkInfo *blkInfo = blocks_[blk];
      if (!blk->initialized() && !allocated[blkInfo->id_]) {
        allocated[blkInfo->id_] = true;
        current_mem += blk->size();
      }
    }

    for (auto &it : curNode->out_edges_) {
      Block *blk = it->blk_;
      BlkInfo *blkInfo = blocks_[blk];
      if (!blk->initialized() && !allocated[blkInfo->id_]) {
        allocated[blkInfo->id_] = true;
        current_mem += blk->size();
      }
    }

    size_t free_mem = 0;
    for (auto &it : free_blocks_[curIndex]) {
      BlkInfo *blkInfo = blocks_[it];
      if (it->initialized() || allocated[blkInfo->id_]) {
        allocated[blkInfo->id_] = false;
        free_mem += it->size();
      }
    }

    if (current_mem > peak_mem) {
      peak_mem = current_mem;
      idx = i;
    }
    chart.push_back(current_mem);
    current_mem -= free_mem;
  }

  // get time
  std::vector<float> node_time;
  std::vector<float> remain_in_time;
  std::vector<float> remain_out_time;
  node_time.resize(nodes_.size());
  remain_in_time.resize(nodes_.size());
  remain_out_time.resize(nodes_.size());
  node_time[0] = node_recs_[ids[0]].time_;
  remain_in_time[0] = remain_out_time[0] = node_time[0];
  for (size_t i = 1; i < ids.size(); ++i) {
    float t = node_recs_[ids[i]].time_;
    remain_in_time[i] = t;
    remain_out_time[i] = t;
    node_time[i] += node_time[i - 1] + t;
  }
  for (size_t i = 0; i < chart.size(); ++i) {
    printf("No[%4ld] OP[%4d] Mem[%10ld] Cumulative Time[%f]\n", i, ids[i],
           chart[i], node_time[i]);
  }

  printf("idx[%ld] peak_mem[%ld]\n", idx, peak_mem);

  // find best blocks
  auto comp = [](const SwapItem &left, const SwapItem &right) {
    return left.second < right.second;
  };
  std::priority_queue<SwapItem, std::vector<SwapItem>, decltype(comp)>
      candidate(comp);

  printf("find from %d\n", swap_infos_.size());
  for (size_t i = 0; i < swap_infos_.size(); ++i) {
    auto swap_info = swap_infos_[i];
    int from = id2order[swap_info->swap_out_];
    int to = id2order[swap_info->swap_next_];

    /*
    if (idx >= to || idx <= from) {
      delete swap_info;
      continue;
    }
    */

    size_t size = swap_info->device_blk_->size();
    float in_time = swap_info->in_rec_.time_;
    float out_time = swap_info->out_rec_.time_;

    // float doa = node_time[to - 1] - node_time[from] - in_time - out_time;
    // float aoa = doa >= 0 ? doa * size : doa / size;
    float wdoa = 0;

    int left = from + 1;
    int right = to - 1;
    for (; left < to; ++left) {
      if (node_time[left] - node_time[from] >= in_time) {
        break;
      }
    }
    for (; right > from; --right) {
      if (node_time[to] - node_time[right] >= out_time) {
        break;
      }
    }

    if (left >= right) {
      delete swap_info;
      continue;
    }

    for (; left < right; ++left) {
      wdoa += (node_time[left] - node_time[left - 1]) * chart[left];
    }

    float score = wdoa;
    SwapItem item = std::make_pair(swap_info, score);
    candidate.push(item);
    printf("candidate block[%d]\n", blocks_[swap_info->device_blk_]->id_);
  }
  swap_infos_.clear();
  swap_out_.clear();
  swap_out_.resize(nodes_.size());
  swap_free_.clear();
  swap_free_.resize(nodes_.size());
  swap_in_.clear();
  swap_in_.resize(nodes_.size());
  swap_wait_.clear();
  swap_wait_.resize(nodes_.size());

  printf("candidates [%d]\n", candidate.size());
  while (!candidate.empty()) {
    SwapItem item = candidate.top();
    candidate.pop();

    auto swap_info = item.first;
    int from = id2order_[swap_info->swap_out_];
    int to = id2order_[swap_info->swap_next_];
    size_t size = swap_info->device_blk_->size();
    float in_time = swap_info->in_rec_.time_;
    float out_time = swap_info->out_rec_.time_;

    int left = from + 1;
    int right = to - 1;
    float t1, t2;
    for (t1 = in_time; left < to && t1 > 0; ++left) {
      t1 -= remain_in_time[left];
    }
    for (t2 = out_time; right > from && t2 > 0; --right) {
      t2 -= remain_out_time[right];
    }
    if (t1 > 0 || t2 > 0 || left >= right) {
      delete swap_info;
      continue;
    }

    printf("from[%d] to[%d]\n", from, to);
    printf("left[%d] right[%d]\n", left, right);

    swap_infos_.push_back(swap_info);
    swap_info->swap_free_ = ids_[left];
    swap_info->swap_in_ = ids_[right];
    swap_out_[swap_info->swap_out_].push_back(swap_info);
    swap_free_[swap_info->swap_free_].push_back(swap_info);
    swap_in_[swap_info->swap_in_].push_back(swap_info);
    swap_wait_[swap_info->swap_next_].push_back(swap_info);
    for (size_t i = from + 1; i < left; ++i) {
      float t = remain_in_time[i];
      remain_in_time[i] = std::max(0.0f, t - in_time);
      in_time -= t;
    }
    for (size_t i = to - 1; i > right; --i) {
      float t = remain_out_time[i];
      remain_out_time[i] = std::max(0.0f, t - out_time);
      out_time -= t;
    }

    Block *blk = swap_info->device_blk_;
    BlkInfo *blkInfo = blocks_[blk];
    printf(
        "swap Block[%d] size[%d] score[%f] out[%d] free[%d] in[%d] next[%d]\n",
        blkInfo->id_, blk->size(), item.second, swap_info->swap_out_,
        swap_info->swap_free_, swap_info->swap_in_, swap_info->swap_next_);
    printf("out[%d] free[%d] in[%d] next[%d]\n", id2order[swap_info->swap_out_],
           id2order[swap_info->swap_free_], id2order[swap_info->swap_in_],
           id2order[swap_info->swap_next_]);
  }
}

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

  swap_free_.clear();
  swap_free_.resize(nodes_.size());

  swap_wait_.clear();
  swap_wait_.resize(nodes_.size());

  for (size_t i = 0; i < swap_infos_.size(); ++i) {
    delete swap_infos_[i];
  }
  swap_infos_.clear();

  host_blks_.resize(blocks_.size(), nullptr);
  for (auto it : blocks_) {
    Block *blk = it.first;
    BlkInfo *blkInfo = it.second;
    Block *host_blk = host_blks_[blkInfo->id_];
    if (!host_blk) {
      void *ptr = nullptr;
      // CUDA_CHECK(cudaHostAlloc((void **)&ptr, blk->size(),
      // ptr = malloc(blk->size());
      CUDA_CHECK(cudaMallocHost((void **)&ptr, blk->size()));
      host_blk = new Block(ptr, blk->size());
      host_blks_[blkInfo->id_] = host_blk;
    }
  }
}

void Graph::RecordTime() {
  size_t size = node_recs_.size();

  if (size > 0) {
    CUDA_CHECK(cudaStreamSynchronize(device_->context(0)->stream));
  }

  float total_time = 0;
  for (size_t i = 0; i < size; ++i) {
    auto &rec = node_recs_[i];
    CUDA_CHECK(cudaEventElapsedTime(&rec.time_, rec.start_, rec.end_));
    // printf("OP[%ld] elapsedTime[%f]\n", i, rec.time_);
    total_time += rec.time_;
  }
  printf("total_time[%f]\n", total_time);

  float total_in_time = 0;
  float total_out_time = 0;
  for (size_t i = 0; i < swap_infos_.size(); ++i) {
    auto &in_rec = swap_infos_[i]->in_rec_;
    auto &out_rec = swap_infos_[i]->out_rec_;

    CUDA_CHECK(cudaEventElapsedTime(&in_rec.time_, in_rec.start_, in_rec.end_));
    CUDA_CHECK(
        cudaEventElapsedTime(&out_rec.time_, out_rec.start_, out_rec.end_));

    // Block *blk = swap_infos_[i]->device_blk_;
    // int id = blocks_[blk]->id_;
    // printf("SwapIn: Block[%d] Size[%ld] Time[%f]\n", id, blk->size(),
    //        in_rec.time_);
    // printf("SwapOut: Block[%d] Size[%ld] Time[%f]\n", id, blk->size(),
    //        out_rec.time_);

    total_in_time += in_rec.time_;
    total_out_time += out_rec.time_;
  }

  printf("total_in_time[%f] total_out_time[%f]\n", total_in_time,
         total_out_time);
}

void Graph::ThreadLoop() {
  SwapInfo *swap_info = nullptr;
  for (;;) {
    if (free_queue_.TryPop(swap_info)) {
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

    // wait for the data block to be swapped out
    // printf("in stream wait Swap Out End Event block[%d]\n",
    // blocks_[device_blk]->id_);
    CUDA_CHECK(cudaStreamWaitEvent(in_stream_, swap_info->out_rec_.end_, 0));

    if (start_up_) {  // to record the swapping time
      CUDA_CHECK(cudaEventRecord(swap_info->in_rec_.start_, in_stream_));
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src, host_blk->size(),
                               cudaMemcpyHostToDevice, in_stream_));
    CUDA_CHECK(cudaEventRecord(swap_info->in_rec_.end_, in_stream_));
    // printf("Record Swap In End Event block[%d]\n", blocks_[device_blk]->id_);
  } else {
    swap_info->on_device_ = false;
    const void *src = device_blk->data();
    void *dst = host_blk->mutable_data();

    // wait the operation to complete before swapping the block out
    // printf("out stream wait End Event Op[%d]\n", swap_info->swap_out_);
    CUDA_CHECK(cudaStreamWaitEvent(out_stream_,
                                   node_recs_[swap_info->swap_out_].end_, 0));

    // swap the block out of the device
    if (start_up_) {
      CUDA_CHECK(cudaEventRecord(swap_info->out_rec_.start_, out_stream_));
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src, device_blk->size(),
                               cudaMemcpyDeviceToHost, out_stream_));
    CUDA_CHECK(cudaEventRecord(swap_info->out_rec_.end_, out_stream_));
    // printf("Record Swap Out End Event block[%d]\n",
    // blocks_[device_blk]->id_);

    /*
    // free data if the swapping is complete
    CBData *cb_data = new CBData(this, swap_info);
    cudaStreamAddCallback(out_stream_, Graph::Callback, (void *)(cb_data), 0);
    */
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
