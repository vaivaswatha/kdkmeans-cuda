#ifndef KD_H
#define KD_H

#include "common.h"

#if defined (CPU_KD) || defined (GPU_KD)
typedef struct KdTree {
    int clusterId; // Actual data
    int splitPlane;
    int leftChildIndex;
    int rightChildIndex;
    int parentIndex;
} KdTree; 

// Builds a KD tree and returns a reference to it.
// Note, the tree would be an array of KdTree nodes
// with noOfNodes elements. So hence the tree can be
// copied easily.
// Uses previously allocated memory for KdTree [] unless
// ReAlloc is set to true
KdTree *BuildKdTree(Cluster *nodes, int noOfNodes, bool ReAlloc);

#ifdef CPU_KD
// Returns a pointer to the cluster that is nearest to the given point.
Cluster *NearestNeighbour(KdTree *kdTree, Cluster *nodes, Point point);
#endif

#ifdef GPU_KD
__device__ Cluster *NearestNeighbourGPU(KdTree *kdTree, Cluster *nodes, char *visited, Point point, int noOfNodes);
#endif

#endif // defined (CPU_KD) || defined (GPU_KD)

#endif // KD_H
