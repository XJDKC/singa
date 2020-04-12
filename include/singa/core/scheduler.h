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

#ifndef SINGA_CORE_SCHEDULER_H_
#define SINGA_CORE_SCHEDULER_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "singa/core/common.h"
#include "singa/utils/safe_queue.h"

using std::function;
using std::unordered_map;
using std::vector;

namespace singa {

class Node;
class Edge;
class OpRec;
class Graph;
class Device;
class BlkInfo;
class SwapInfo;

typedef std::vector<OpRec> OpRecVec;
typedef std::vector<Node *> NodeVec;
typedef std::vector<Edge *> EdgeVec;
typedef std::vector<Block *> BlockVec;
typedef std::vector<SwapInfo *> SwapInfoVec;
typedef std::function<void(Context *)> OpFunc;
typedef std::unordered_map<Block *, BlkInfo *> Blk2InfoMap;

enum BlockType { kUnknow, kInput, kParam, kInter, kEnd };

class Node {
 public:
  Node(int id, OpFunc &&op) : id_(id), op_(std::move(op)) {}

  void AddInEdge(Edge *in_edge);
  void AddOutEdge(Edge *out_edge);

  // getters of Node
  int id() const { return id_; }
  const EdgeVec &in_edges() const { return in_edges_; }
  const EdgeVec &out_edges() const { return out_edges_; }

 private:
  friend Graph;

  int id_;
  OpFunc op_;
  EdgeVec in_edges_;
  EdgeVec out_edges_;
};

class Edge {
 public:
  Edge(int id, Block *blk, Node *src_node, Node *dst_node)
      : id_(id), blk_(blk), src_node_(src_node), dst_node_(dst_node) {}

  void SetBlock(Block *blk);
  void SetSrcNode(Node *src_node);
  void SetDstNode(Node *dst_node);

  // getters of Edge
  int id() const { return id_; }
  Block *block() const { return blk_; }
  Node *src_node() const { return src_node_; }
  Node *dst_node() const { return dst_node_; }

 private:
  friend Graph;

  int id_;
  Block *blk_;
  Node *src_node_;
  Node *dst_node_;
};

class BlkInfo {
 public:
  BlkInfo(int id, Block *blk, BlockType type = BlockType::kUnknow)
      : id_(id),
        blk_(blk),
        type_(type),
        graph_ref_(0),
        write_edge_(nullptr),
        write_node_(nullptr) {}

  // getters of BlkInfo
  int id() const { return id_; }
  Block *block() const { return blk_; }
  BlockType type() const { return type_; }
  int graph_ref() const { return graph_ref_; }
  Edge *write_edge() const { return write_edge_; }
  Node *write_node() const { return write_node_; }
  const NodeVec &used_nodes() const { return used_nodes_; }
  Node *used_node(const size_t idx) const;

 private:
  friend Graph;

  int id_;
  Block *blk_;
  BlockType type_;
  int graph_ref_;
  Edge *write_edge_;    // the edge of last node to write data into blk
  Node *write_node_;    // last node that writes the block
  NodeVec used_nodes_;  // the nodes that use this block
};

class OpRec {
 public:
  OpRec();

 private:
  friend Graph;

  float time_;
  cudaEvent_t start_;
  cudaEvent_t end_;
};

class SwapInfo {
 public:
  SwapInfo()
      : next_(-1),
        swap_in_(-1),
        swap_out_(-1),
        host_blk_(nullptr),
        device_blk_(nullptr) {}
  SwapInfo(int next, int swap_in, int swap_out, Block *host_blk,
           Block *device_blk)
      : next_(next),
        swap_in_(swap_in),
        swap_out_(swap_out),
        host_blk_(host_blk),
        device_blk_(device_blk) {}

 private:
  friend Graph;

  bool on_device_ = true;
  int next_;      // next node id
  int swap_in_;   // swap in node id
  int swap_out_;  // swap out node id
  Block *host_blk_;
  Block *device_blk_;
  OpRec in_rec_;
  OpRec out_rec_;
  mutable std::mutex mtx_;
};

class Graph {
 public:
  struct CBData {
    Graph *graph_;
    SwapInfo *swap_info_;

    CBData(Graph *graph, SwapInfo *swap_info)
        : graph_(graph), swap_info_(swap_info) {}
  };

  ~Graph();
  Graph(Device *device);

  void Reset();
  void Debug();
  void RunGraph();
  void RunInSerial();
  void AddOperation(OpFunc &&op, const BlockVec &read_blocks,
                    const BlockVec &write_blocks);

  // getters of Graph
  const NodeVec &nodes() const { return nodes_; }
  const EdgeVec &edges() const { return edges_; }
  const Blk2InfoMap &blocks() const { return blocks_; }

  const BlockVec &write_blocks() const { return write_blocks_; }

  bool dirty() const { return dirty_; }
  const NodeVec &begin_nodes() const { return begin_nodes_; }
  const std::vector<NodeVec> &next_nodes() const { return next_nodes_; }
  const std::vector<BlockVec> &free_blocks() const { return free_blocks_; }

  Node *node(const size_t idx) const;
  Edge *edge(const size_t idx) const;
  BlkInfo *block(Block *blk) const;

  Block *write_block(const size_t idx) const;

  Node *begin_node(const size_t idx) const;
  const NodeVec &next_nodes(const size_t idx) const;
  const BlockVec &free_blocks(const size_t idx) const;

 private:
  void Analysis();
  void AutoSwap();
  void FreeLoop();
  void ResetPlan();
  void RecordTime();
  void ReserveMem(size_t size);
  void SwapBlock(SwapInfo *swap_info, bool direct);
  void AddSyncOp(function<void(Context *)> &&op);

  static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status,
                                 void *data);

 private:
  Device *device_;

  // nodes, edges and blocks included in the calculation graph
  NodeVec nodes_;
  EdgeVec edges_;
  Blk2InfoMap blocks_;

  // Blocks written by the last operation, used for sync op
  BlockVec write_blocks_;

  // Calculation graph analysis
  bool dirty_ = false;
  NodeVec begin_nodes_;
  std::vector<NodeVec> next_nodes_;
  std::vector<BlockVec> free_blocks_;

  // AutoSwap
  bool autoswap_ = false;
  bool start_up_ = true;
  size_t threshold_ = 1048576;  // 1048576 = 1MB
  cudaStream_t swap_;
  BlockVec host_blks_;
  OpRecVec node_recs_;
  SwapInfoVec swap_infos_;
  std::vector<SwapInfoVec> swap_in_;
  std::vector<SwapInfoVec> swap_out_;
  std::vector<SwapInfoVec> swap_wait_;

  // Free blocks in callback functions
  std::thread thread_;
  SafeQueue<SwapInfo *> free_queue_;
};

/// Scheduling Tensor operations with dependency detection.
class Scheduler {};

}  // namespace singa
#endif  // SINGA_CORE_SCHEDULER_H_
