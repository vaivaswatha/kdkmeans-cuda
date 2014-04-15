#ifndef KD_C
#define KD_C

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "kd.h"
#include "common.h"

#if defined(CPU_KD) || defined (GPU_KD)

static KdTree *KdTreeMemPool = NULL;
// Below array is sorted, rather than actual points.
static int *clusterIdArr = NULL;
struct SortWorkList {
    int from;
    int to;
} *sortWorkList = NULL;

// Number of elements in previous kd tree built.
static int memAllocated = 0;
static int nextFreeNode;

static void QuickSort(const Cluster *nodes, int *idArr, int fFrom, int fTo, int axis) {
    
    int from, to, workListTop, nextEmptySlot, i, temp;

    assert(sortWorkList);
    workListTop = 0;

    sortWorkList[workListTop].from = fFrom;
    sortWorkList[workListTop].to = fTo;

    while (workListTop >= 0) {
        from = sortWorkList[workListTop].from;
        to = sortWorkList[workListTop].to;
        workListTop--;
        // TO is always the pivot.

        assert(from < to);
        nextEmptySlot = from;
        for (i = from; i < to; i++) {
            if (nodes[idArr[i]].pt.loc[axis] <= nodes[idArr[to]].pt.loc[axis]) {
                temp = idArr[nextEmptySlot];
                idArr[nextEmptySlot] = idArr[i];
                idArr[i] = temp;
                nextEmptySlot++;
            }
        }
        temp = idArr[to];
        idArr[to] = idArr[nextEmptySlot];
        idArr[nextEmptySlot] = temp;

        // idArr[nextEmptySlot] now contains "to", the pivot.
        if (from < nextEmptySlot-1) {
            workListTop++;
            sortWorkList[workListTop].from = from;
            sortWorkList[workListTop].to = nextEmptySlot-1;
        }
        if (nextEmptySlot+1 < to) {
            workListTop++;
            sortWorkList[workListTop].from = nextEmptySlot+1;
            sortWorkList[workListTop].to = to;
        }
    }
   
#ifdef DEBUG
    for (i = fFrom; i <= fTo; i++) {
        if (i < fTo) {
            assert(nodes[idArr[i]].pt.loc[axis] <= nodes[idArr[i+1]].pt.loc[axis]);
        }
    }
#endif
}

static int BuildKdTreeRec(Cluster *nodes, int *idArr, int from, int to, int depth) {
    
    KdTree *currentNode, *childNode;
    int currentNodeIndex, median;

    if (from == to) {
        currentNodeIndex = nextFreeNode++;
        currentNode = &KdTreeMemPool[currentNodeIndex];
        currentNode->splitPlane = depth%DIMENSIONS;
        currentNode->clusterId = idArr[from];
        currentNode->leftChildIndex = -1;
        currentNode->rightChildIndex = -1;
        return currentNodeIndex;
    }

    QuickSort(nodes, idArr, from, to, depth%DIMENSIONS);

    currentNodeIndex = nextFreeNode++;
    currentNode = &KdTreeMemPool[currentNodeIndex];
    currentNode->splitPlane = depth%DIMENSIONS;
    // Current node will be the median
    median = (from+to)/2;
    currentNode->clusterId = idArr[median];

    if (median != from) {
        currentNode->leftChildIndex = BuildKdTreeRec(nodes, idArr, from, median-1, depth+1);
    } else {
        currentNode->leftChildIndex = -1;
    }
    currentNode->rightChildIndex = BuildKdTreeRec(nodes, idArr, median+1, to, depth+1);

    if (currentNode->leftChildIndex != -1) {
        childNode = &KdTreeMemPool[currentNode->leftChildIndex];
        childNode->parentIndex = currentNodeIndex;
    }
    assert(currentNode->rightChildIndex != -1);
    childNode = &KdTreeMemPool[currentNode->rightChildIndex];
    childNode->parentIndex = currentNodeIndex;


    return currentNodeIndex;
}

#ifdef DEBUG
void DumpKdTree(Cluster *nodes, KdTree *tree) {

    printf("%d [label=%c]\n", tree->clusterId, 
           (tree->splitPlane == X_AXIS) ? 'X' : 'Y');
    if(tree->leftChildIndex != -1) {
        printf("%d -> %d\n", tree->clusterId,
               KdTreeMemPool[tree->leftChildIndex].clusterId);
        DumpKdTree(nodes, &KdTreeMemPool[tree->leftChildIndex]);
    }
    if (tree->rightChildIndex != -1) {
        printf("%d -> %d\n", tree->clusterId, 
               KdTreeMemPool[tree->rightChildIndex].clusterId);
        DumpKdTree(nodes, &KdTreeMemPool[tree->rightChildIndex]);
    }

}
#endif

KdTree *BuildKdTree(Cluster *nodes, int noOfNodes, bool ReAlloc) {

    KdTree *root;
    int i, rootIndex;

    if (memAllocated < noOfNodes || ReAlloc) {
        if (memAllocated > 0) {
            if (!ReAlloc) {
                free(KdTreeMemPool);
            }
            free(clusterIdArr);
            free(sortWorkList);
        }
        KdTreeMemPool = (KdTree *) malloc (sizeof(KdTree)*noOfNodes);
        clusterIdArr = (int *) malloc (sizeof(int)*noOfNodes);
        sortWorkList = (struct SortWorkList *) malloc (sizeof(struct SortWorkList)*noOfNodes);
        memAllocated = noOfNodes;
    }
    
    for (i = 0; i < noOfNodes; i++) {
        clusterIdArr[i] = i;
        assert(clusterIdArr[i] == nodes[i].pt.clusterId);
    }

    nextFreeNode = 0;
    rootIndex = BuildKdTreeRec(nodes, clusterIdArr, 0, noOfNodes-1, 0);
    root = &KdTreeMemPool[rootIndex];
    root->parentIndex = -1;

#ifdef DEBUG
    printf("digraph KdTree {\n");
    DumpKdTree(nodes, root);
    printf("}\n");
#endif

    return root;
}
#endif // defined(CPU_KD) || defined (GPU_KD)

#ifdef GPU_KD
__device__ Cluster *NearestNeighbourGPU(KdTree *kdTree, Cluster *nodes, char *visited, Point point, int noOfNodes) {

    int cursor, prevCursor, axis, bestNode;
    double bestDist, subRoot, dist;
    KdTree *KdTreeMemPool = kdTree;

    // assert(visited);
    memset(visited, 0, sizeof(char)*noOfNodes);
    bestDist = -1.0;
    bestNode = -1;

    subRoot = 0;

    // cursor, prevCursor can be cosidered pointers to KdTree nodes.
    // KdTreeMemPool[cursor] is an indirection to that node.
    // KdTreeMemPool[cursor].clusterId is a pointer to the cluster (centroid) that the 
    // node represents. nodes[KdTreeMemPool[cursor].clusterId] is an indrection into the
    // actual cluster node.

    while (1) {

        prevCursor = -1;
        cursor = subRoot;
        
        // Traverse down the binary (kd) tree to the nearest leaf.
        while (cursor != -1) {
            axis = KdTreeMemPool[cursor].splitPlane;
            prevCursor = cursor;
            // TODO: Perf- possibly eliminate below divergence.
            // if point(x/y) <= cursor(x/y), take left path.
            if (point.loc[axis] <= nodes[KdTreeMemPool[cursor].clusterId].pt.loc[axis]) {
                cursor = KdTreeMemPool[cursor].leftChildIndex;
            } else {
                // else right path.
                cursor = KdTreeMemPool[cursor].rightChildIndex;
            }
        }

        // Now unwind the stack
        cursor = prevCursor; // cursor now points to the leaf node reached. 
        prevCursor = -1;
        while (cursor != -1) {
            if (!visited[cursor]) {
                visited[cursor] = 1;
                dist = GetDistanceGPU(point, nodes[KdTreeMemPool[cursor].clusterId].pt);
                if (dist < bestDist || bestNode == -1) {
                    bestNode = cursor;
                    bestDist = dist;
                }

                // See if sibling subtree needs to be visited
                axis = KdTreeMemPool[cursor].splitPlane;
                dist = MOD(nodes[KdTreeMemPool[cursor].clusterId].pt.loc[axis]-point.loc[axis]);
                if (dist < bestDist) {
                    // Traverse down the the other subtree.
                    if (KdTreeMemPool[cursor].leftChildIndex == prevCursor && 
                        KdTreeMemPool[cursor].rightChildIndex != -1)
                    {
                        subRoot = KdTreeMemPool[cursor].rightChildIndex;
                        break;
                    } else if (KdTreeMemPool[cursor].leftChildIndex != -1) {
                        subRoot = KdTreeMemPool[cursor].leftChildIndex;
                        break;
                    }
                }
            }
            // cursor = parent(cursor)
            prevCursor = cursor;
            cursor = KdTreeMemPool[cursor].parentIndex;
        }
        if (cursor == -1)
            break;
        // Else, traverse down new subRoot and up again.
    }
    // assert(bestNode != -1 && bestDist >= 0);
    return &nodes[KdTreeMemPool[bestNode].clusterId];
}
#endif // GPU_KD

#ifdef CPU_KD
Cluster *NearestNeighbour(KdTree *kdTree, Cluster *nodes, Point point) {

    int cursor, prevCursor, axis, bestNode;
    double bestDist, subRoot, dist;
    KdTree *KdTreeMemPool = kdTree;
    int *visited = clusterIdArr;

    assert(visited);
    memset(visited, 0, sizeof(int)*memAllocated);
    bestDist = -1.0;
    bestNode = -1;

    subRoot = 0;

    // cursor, prevCursor can be cosidered pointers to KdTree nodes.
    // KdTreeMemPool[cursor] is an indirection to that node.
    // KdTreeMemPool[cursor].clusterId is a pointer to the cluster (centroid) that the 
    // node represents. nodes[KdTreeMemPool[cursor].clusterId] is an indrection into the
    // actual cluster node.

    while (1) {

        prevCursor = -1;
        cursor = subRoot;
        
        // Traverse down the binary (kd) tree to the nearest leaf.
        while (cursor != -1) {
            axis = KdTreeMemPool[cursor].splitPlane;
            prevCursor = cursor;
            // TODO: Perf- possibly eliminate below divergence.
            // if point(x/y) <= cursor(x/y), take left path.
            if (point.loc[axis] <= nodes[KdTreeMemPool[cursor].clusterId].pt.loc[axis]) {
                cursor = KdTreeMemPool[cursor].leftChildIndex;
            } else {
                // else right path.
                cursor = KdTreeMemPool[cursor].rightChildIndex;
            }
        }

        // Now unwind the stack
        cursor = prevCursor; // cursor now points to the leaf node reached. 
        prevCursor = -1;
        while (cursor != -1) {
            if (!visited[cursor]) {
                visited[cursor] = 1;
                dist = GetDistance(point, nodes[KdTreeMemPool[cursor].clusterId].pt);
                if (dist < bestDist || bestNode == -1) {
                    bestNode = cursor;
                    bestDist = dist;
                }

                // See if sibling subtree needs to be visited
                axis = KdTreeMemPool[cursor].splitPlane;
                dist = MOD(nodes[KdTreeMemPool[cursor].clusterId].pt.loc[axis]-point.loc[axis]);
                if (dist < bestDist) {
                    // Traverse down the the other subtree.
                    if (KdTreeMemPool[cursor].leftChildIndex == prevCursor && 
                        KdTreeMemPool[cursor].rightChildIndex != -1)
                    {
                        subRoot = KdTreeMemPool[cursor].rightChildIndex;
                        break;
                    } else if (KdTreeMemPool[cursor].leftChildIndex != -1) {
                        subRoot = KdTreeMemPool[cursor].leftChildIndex;
                        break;
                    }
                }
            }
            // cursor = parent(cursor)
            prevCursor = cursor;
            cursor = KdTreeMemPool[cursor].parentIndex;
        }
        if (cursor == -1)
            break;
        // Else, traverse down new subRoot and up again.
    }
    assert(bestNode != -1 && bestDist >= 0);
    return &nodes[KdTreeMemPool[bestNode].clusterId];
}
#endif // CPU_KD

#endif // KD_C
