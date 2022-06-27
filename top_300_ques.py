import collections
from cmath import inf
from copy import deepcopy
from typing import List, Optional, final
import random
import functools  

"""链表节点以及链表建立"""
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    def __repr__(self) -> str:
        prt = [self.val]
        next = self.next
        while 1:
            if next != None:
                prt.append(next.val)
                next = next.next
            else:
                break
        return str(prt)

class LinkList(object):
    def __init__(self):
        self.head = None
    def initList(self, data):
        if len(data) == 0: return None
        self.head = ListNode(data[0])
        p = self.head
        for i in data[1:]:
            node = ListNode(i)
            p.next = node
            p = p.next
        return self.head

"""二叉树节点以及二叉树建立"""
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self) -> str:
        return '%s：(left->%s right->%s)'%(self.val, self.left, self.right)

"""必须都以None为叶子节点的形式，以层级顺序构建list输入，返回根节点"""
def bulid_Tree(data_list: List) -> TreeNode:
    if not data_list:
        return None
    root = TreeNode(int(data_list[0]))
    queue = collections.deque([root])
    i = 1
    while queue:
        node = queue.popleft()
        if data_list[i] != None:
            node.left = TreeNode(data_list[i])
            queue.append(node.left)
        i += 1
        if data_list[i] != None:
            node.right = TreeNode(data_list[i])
            queue.append(node.right)
        i += 1
    return root

"""带权重的并查集"""
class UnionFind:        
    def __init__(self):
        """记录每个节点的父节点、记录每个节点到根节点的权重"""
        self.father = {}
        self.value = {}
    
    def find(self,x):
        """查找根节点、路径压缩、更新权重"""
        root = x
        # 节点更新权重的时候要放大的倍数
        base = 1
        while self.father[root] != None:
            root = self.father[root]
            base *= self.value[root]
        
        while x != root:
            original_father = self.father[x]
            ##### 离根节点越远，放大的倍数越高
            self.value[x] *= base
            base /= self.value[original_father]
            self.father[x] = root
            x = original_father
         
        return root
    
    def merge(self,x,y,val):
        """合并两个节点"""
        root_x,root_y = self.find(x),self.find(y)
        
        if root_x != root_y:
            self.father[root_x] = root_y
            ##### 四边形法则更新根节点的权重
            self.value[root_x] = self.value[y] * val / self.value[x]

    def is_connected(self,x,y):
        """两节点是否相连"""
        return x in self.value and y in self.value and self.find(x) == self.find(y)
    
    def add(self,x):
        """添加新节点，初始化权重为1.0"""
        if x not in self.father:
            self.father[x] = None
            self.value[x] = 1.0

"""_________________________________________________Top_300_questions_________________________________________________________________"""

"""1."""


