from cmath import inf
from copy import deepcopy
from hashlib import new
from inspect import stack
from turtle import left, right
from typing import List, Optional, final


"""1.两数之和"""
# class Solution:
#     # def twoSum(self, nums: List[int], target: int) -> List[int]:
#     #     count_dele = 0
#     #     for i in range(len(nums)):
#     #         j = target - nums[0]
#     #         nums.remove(nums[0])
#     #         count_dele += 1
#     #         if j in nums:
#     #             j_index = nums.index(j)
#     #             return [count_dele-1,j_index+count_dele]
#     def twoSum(self, nums, target):   #通过构建额外字典的方式稍微增加内存加快速度，python中用字典查找很快。 列表读取为O(1)，查找为O(n)，而字典查找为O(1)，字典本质是散列表
#         hashmap={}                    #散列值存在的两个条件：__hash__、__eq__,有这两个方法基本为不可变对象如str、int. 散列表查找时即为哈希查找，如字典则对查找key值用hash函数则可直接查找
#         for ind,num in enumerate(nums):
#             hashmap[num] = ind
#         for i,num in enumerate(nums):
#             j = hashmap.get(target - num)
#             if j is not None and i!=j:
#                 return [i,j]

"""2.两数相加(预先定义好了单链表)"""
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

# class Solution:
#     def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
#         head_node = ListNode(l1.val + l2.val)
#         tp = head_node
#         while (l1 and (l1.next != None)) or (l2 and (l2.next != None)) or (tp.val > 9):      #把None视为0进行不断迭代就可以省去多个判断
#             l1, l2 = l1.next if l1 else l1, l2.next if l2 else l2
#             tmpsum = (l1.val if l1 else 0) + (l2.val if l2 else 0)
#             tp.next = ListNode(tp.val//10 + tmpsum)     #没有单独存储进位状态而是直接通过//操作
#             tp.val %= 10
#             tp = tp.next
#         return head_node
# example = Solution()
# print(example.addTwoNumbers(LinkList().initList([8,4,3]),LinkList().initList([5,6,4])))

"""3. 无重复字符的最长子串"""
# class Solution:
#     # def lengthOfLongestSubstring(self, s: str) -> int:     #mine
#     #     max_length = 0
#     #     i,j = 0,0
#     #     while (j<len(s)):                                  #我的每次移动的是右指针
#     #         j += 1
#     #         if len(set(s[i:j+1])) != j-i+1:                #这一步效率不高
#     #             i += 1
#     #         max_length = max(max_length,j-i+1)
#     #     return max_length

#     def lengthOfLongestSubstring(self, s: str) -> int:     #官方实现，滑动窗口
#         occ = set()
#         n = len(s)
#         rk, ans = -1, 0
#         for i in range(n):
#             if i != 0:
#                 occ.remove(s[i - 1])
#             while rk + 1 < n and s[rk + 1] not in occ:
#                 occ.add(s[rk + 1])
#                 rk += 1
#             ans = max(ans, rk - i + 1)
#         return ans

"""4. 寻找两个正序数组的中位数"""
#略，主要思路是要求时间复杂度为log级，所以需要用到二分查找的递归方式。
#但是我选择合并+sort

"""5. 最长回文子串"""      #动态规划，状态、转移方程、边界条件，本题的状态转移方程：P(i,j)=P(i+1,j−1)∧(Si​==Sj​)
# class Solution:
#     def longestPalindrome(self, s: str) -> str:
#         n = len(s)
#         max_length = 1
#         begin = 0
#         if n<2: return s
#         dp = [[False]*n for _ in range(n)]
#         #确定边界条件
#         dp[n-1][n-1] = True
#         for i in range(n-1):
#             dp[i][i] = True
#             if s[i] == s[i+1]:
#                 dp[i][i+1] = True  #我的代码在这里有点繁琐了，但思想在就行
#                 max_length = 2
#                 begin = i
#         #动态规划           
#         for length in range(2,n+1):
#             for i in range(n-1):
#                 j = length+i-1
#                 if j>=n:
#                     break
#                 if s[i] != s[j]:
#                     dp[i][j] = False
#                 elif i!=j and j!=i+1:
#                     dp[i][j] = dp[i+1][j-1]
#                 if dp[i][j] and j-i+1 > max_length:
#                     max_length = j-i+1
#                     begin = i
#         return s[begin:max_length+begin]

"""10. 正则表达式匹配"""    #动态规划 选择从右往左扫描 状态:dp[i][j]表示s的前i个字符与p的前j个字符是否匹配 转移方程：多种情况  边界：空串的情况
# class Solution:   
#     def isMatch(self, s: str, p: str) -> bool:
#         n,m = len(s), len(p)
#         #先判断有空串的情况  其实不需要，因为矩阵大小为(n+1)*(m+1)
#         # if n==0 and m==0: return True
#         # if m==0 : return False
#         # if n==0 : return set(s[1::2])==set('*') and len(s)%2 == 0

#         dp = [[False]*(m+1) for _ in range(n+1)]  #需要注意的是矩阵的尺寸是(n+1)*(m+1)
#         #边界条件
#         for i in range(n+1):
#             dp[i][0] = False
#         dp[0][0] = True
#         for j in range(1,m+1):
#             if p[j-1] == '*':
#                 dp[0][j] = dp[0][j-2]
#         #状态转移
#         for i in range(1,n+1):
#             for j in range(1,m+1):
#                 if s[i-1] == p[j-1] or p[j-1] == '.':
#                     dp[i][j] = dp[i-1][j-1]
#                 else:
#                     if p[j-1] == '*':
#                         if s[i-1] == p[j-2] or p[j-2] == '.':
#                             dp[i][j] = dp[i][j-2] or dp[i-1][j-2] or dp[i-1][j]
#                         else:
#                             dp[i][j] = dp[i][j-2]
#                     else:
#                         dp[i][j] = False
#         return dp[n][m]

"""11. 盛最多水的容器"""  #经典面试题，最优做法是双指针
# class Solution:
#     def maxArea(self, height: List[int]) -> int:
#         n = len(height)
#         if n<2: return 0
#         max_area = 0
#         i,j = 0,n-1
#         while i!=j:
#             area = min(height[i], height[j])*(j-i)
#             max_area = max(max_area, area)
#             if height[i] <= height[j]:
#                 i += 1
#             else:
#                 j -= 1
#         return max_area

"""15. 三数之和"""   #也是双指针，思想在于a固定，b增大时c减小，就可以做出一个双向的指针来减少时间复杂度。所以双指针的设计就是双向往中间移动来减少复杂度
# class Solution:
#     # def threeSum(self, nums: List[int]) -> List[List[int]]:   #这是我用二数之和的hashmap思想做的O(n²)的答案,还需改进
#     #     n = len(nums)
#     #     if n<3: return []
#     #     res = []
#     #     hashmap = {}
#     #     for index,num in enumerate(nums):
#     #         hashmap[num] = index
#     #     for i in range(n-1):
#     #         for j in range(i+1,n):
#     #             k = hashmap.get(-nums[i]-nums[j])
#     #             if k is not None and k>j and sorted([nums[i],nums[j],nums[k]]) not in res:
#     #                 res.append(sorted([nums[i],nums[j],nums[k]]))
#     #     return res
#     def threeSum(self, nums: List[int]) -> List[List[int]]:
#         n = len(nums)
#         nums.sort()   #nlogn
#         ans = list()

#         for first in range(n):
#             if first>0 and nums[first-1] == nums[first]:
#                 continue
#             third = n-1
#             target = -nums[first]
#             for second in range(first+1,n):
#                 if second>first+1 and nums[second-1] == nums[second]:
#                     continue
#                 while third > second and nums[third] + nums[second] > target:
#                     third -= 1
#                 if third == second:
#                     break
#                 if nums[third] + nums[second] == target:
#                     ans.append([nums[first], nums[second], nums[third]])
#         return ans
        
"""17. 电话号码的字母组合"""   #只要看到所有组合这种字眼，基本都是广度优先搜索或者深度优先搜索算法了，广度优先搜索算法基本用的就是队列，深度优先搜索算法用的基本都是递归。
# # class Solution:  
# #     def letterCombinations(self, digits: str) -> List[str]:
# #         if len(digits) == 0: return []
# #         nums_to_letter = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
# #         ans = ['']
# #         for i in digits:
# #             new_ans = []
# #             for j in nums_to_letter[i]:
# #                 for value in ans:
# #                     new_ans.append(value+j)    
# #             ans = new_ans
# #         return ans
# class Solution:
#     def letterCombinations(self, digits: str) -> List[str]:
#         if not digits:
#             return list()
        
#         phoneMap = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}

#         def backtrack(index: int):
#             if index == len(digits):
#                 combinations.append("".join(combination))
#             else:
#                 print(combination)
#                 digit = digits[index]
#                 for letter in phoneMap[digit]:
#                     combination.append(letter)
#                     backtrack(index + 1)
#                     combination.pop()

#         combination = list()
#         combinations = list()
#         backtrack(0)
#         return combinations

"""19. 删除链表的倒数第 N 个结点"""  #1.两次遍历的方法  2.用栈后进先出的思想比较好  3.双指针方法，一个指针比另一个指针快n     2. 3.都很秒
# class Solution:
#     def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
#         num = 1
#         temp = head
#         node_dict = {1:head}
#         while temp.next != None:
#             num += 1
#             temp = temp.next
#             node_dict[num] = temp       #我的用字典存储的方法，用到了额外的空间
#         key = num - n
#         if key==0:
#             return node_dict[2] if num>1 else None
#         elif n!=1:
#             node_dict[key].next = node_dict[key+2]
#         else:
#             node_dict[key].next = None
#         return head

"""20. 有效的括号"""  #用栈
# class Solution:
#     def isValid(self, s: str) -> bool:
#         map_dict = {'(':')',')':'','[':']',']':'','{':'}','}':''}
#         stack = []
#         # for kuohao in s:
#         #     if len(stack)==0:
#         #         stack.append(kuohao)
#         #     elif map_dict[stack[-1]] == kuohao:
#         #         stack.pop()
#         #     else:
#         #         stack.append(kuohao)
#         # return True if len(stack)==0 else False
#         for kuohao in s:
#             if len(stack)==0 or map_dict[stack[-1]]!=kuohao:
#                 stack.append(kuohao)
#             else :
#                 stack.pop()
#         return True if len(stack)==0 else False

"""21. 合并两个有序链表"""  #递归！
# class Solution:
#     def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
#         if not list1: return list2
#         if not list2: return list1
#         if list1.val <= list2.val:
#             list1.next = self.mergeTwoLists(list1.next, list2)
#             return list1
#         else:
#             list2.next = self.mergeTwoLists(list1, list2.next)
#             return list2

"""22. 括号生成"""
# class Solution:
#     def generateParenthesis(self, n: int) -> List[str]:
#         res = []
#         cur_str = ''
#         def dfs(cur_str,left,right):
#             if left==0 and right==0:
#                 res.append(cur_str)
#                 return 
#             if right<left:     #剪枝 ,这一步秒啊，之前的dfs似乎都没有涉及到需要剪枝的情况
#                 return
#             if right>0:
#                 dfs(cur_str+')',left,right-1)
#             if left>0:
#                 dfs(cur_str+'(',left-1,right)
#         dfs(cur_str,n,n)
#         return res

"""23.合并K个升序链表"""   #分治合并比顺序合并reduce效率高， 用堆来维护队列进行排序的方法也很不错
# class Solution:
#     def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
#         list = []
#         for i in lists:
#             list.append(LinkList().initList(i))
#         lists = list
#         if not lists:return 
#         n = len(lists)
#         return self.merge(lists, 0, n-1)
#     def merge(self,lists, left, right):
#         if left == right:
#             return lists[left]
#         mid = left + (right - left) // 2
#         l1 = self.merge(lists, left, mid)
#         l2 = self.merge(lists, mid+1, right)
#         return self.mergeTwoLists(l1, l2)
#     def mergeTwoLists(self,l1, l2):
#         if not l1:return l2
#         if not l2:return l1
#         if l1.val < l2.val:
#             l1.next = self.mergeTwoLists(l1.next, l2)
#             return l1
#         else:
#             l2.next = self.mergeTwoLists(l1, l2.next)
#             return l2
# # class Solution:
# #     def mergeKLists(self, lists: List[ListNode]) -> ListNode:
# #         list = []
# #         for i in lists:
# #             list.append(LinkList().initList(i))
# #         lists =list
# #         import heapq
# #         dummy = ListNode(0)
# #         p = dummy
# #         head = []
# #         for i in range(len(lists)):
# #             if lists[i] :
# #                 heapq.heappush(head, (lists[i].val, i))
# #                 print(head)
# #                 lists[i] = lists[i].next
# #         while head:
# #             val, idx = heapq.heappop(head)
# #             p.next = ListNode(val)
# #             p = p.next
# #             print('1')
# #             if lists[idx]:
# #                 heapq.heappush(head, (lists[idx].val, idx))
# #                 lists[idx] = lists[idx].next
# #                 print('2')
# #         return dummy.next

"""31. 下一个排列"""
# class Solution:
#     def nextPermutation(self, nums: List[int]) -> None:
#         n = len(nums)
#         if n<2: return nums
#         for i in range(1,n+1):
#             if nums[-i] > nums[-i-1]:
#                 index = n-i-1
#                 break
#             if i == n-1:
#                 nums.sort()
#                 print(nums)
#                 return
#         for j in range(1,n+1):
#             if nums[-j] > nums[index]:
#                 index2 = n-j
#                 break
#         temp = nums[index2]
#         nums[index2] = nums[index]
#         nums[index] = temp
#         nums[index+1:] = sorted(nums[index+1:])
#         print(nums)
#         return

"""32. 最长有效括号"""   #栈或dp, dp的方法说明了第一步如何选取状态很关键，选好了复杂度难度都会下降。  维度考虑也很关键，比如这里用的一维dp，状态为以i下标结尾子字符串的最大有效长度
# class Solution:
#     def longestValidParentheses(self, s: str) -> int:
#         if not s: return 0
#         n = len(s)
#         dp = [0]*(n+2)   #多两位防止i-dp[i-1]-2  为负        这个操作告诉我在允许改动原数组的情况可以额外延长来处理负索引的情况
#         s = s+'.' #多1位防止i-dp[i-1]-1  为负 
#         for i in range(1,n):
#             if s[i] == '(':
#                 dp[i] = 0
#             elif s[i-1] == '(':
#                 dp[i] = dp[i-2] + 2
#             elif s[i-1] == ')' and s[i-dp[i-1]-1] == '(':
#                 dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
#         return max(dp)

"""33. 搜索旋转排序数组"""  #log(n)要求即为二分查找（前提是有序）
# class Solution:
#     def search(self, nums: List[int], target: int) -> int:
#         low,high,mid = 0,len(nums)-1,0
#         while (low <= high):
#             mid = (high + low) // 2
#             if (nums[mid] == target):
#                 return mid
#             if nums[low] <= nums[mid]:
#                 if nums[low] <= target < nums[mid]:
#                     high = mid - 1
#                 else:
#                     low = mid +1
#             else:
#                 if nums[mid] < target <= nums[high]:
#                     low = mid + 1
#                 else:
#                     high = mid - 1
#         return -1

"""34. 在排序数组中查找元素的第一个和最后一个位置"""   #二分的思想并不难，但细节很多  本题就是用到二分查找边界的思想
# class Solution:
#     def leftMargin(self, nums: List[int], target: int):
#         low, high = 0, len(nums) - 1
#         while low <= high:
#             mid = low + (high - low) // 2
#             if nums[mid] == target:
#                 high = mid - 1
#             elif nums[mid] > target:
#                 high = mid - 1
#             else:
#                 low = mid + 1
#         if nums[low] == target:
#             return low
#         else:
#             return -1
#     def rightMargin(self, nums: List[int], target: int):
#         low, high = 0, len(nums) - 1
#         while low <= high:
#             mid = low + (high - low) // 2
#             if nums[mid] == target:
#                 low = mid + 1
#             elif nums[mid] > target:
#                 high = mid - 1
#             else:
#                 low = mid + 1
#         if nums[high] == target:
#             return high
#         else:
#             return -1
#     def searchRange(self, nums: List[int], target: int) -> List[int]:
#         if len(nums) == 0 or nums[0] > target or nums[-1] < target:
#             return [-1,-1]
#         lm = self.leftMargin(nums, target)
#         rm = self.rightMargin(nums, target)
#         return [lm,rm]

"""39. 组合总和"""  #看见组合总和就晓得是回溯了
# class Solution:
#     def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
#         res = []
#         cur_ans = []
#         def dfs(List, target, cur_ans):
#             if target == 0:
#                 res.append(cur_ans) 
#                 return
#             if target > 0:
#                 for index,i in enumerate(List):
#                     dfs(List[index:], target-i, cur_ans+[i])
#         dfs(candidates,target,cur_ans)
#         return res

"""42. 接雨水"""  #此题还是应该先推导出来下标i处能接的雨水量    1.动态规划   2.单调栈   3.双指针
# class Solution:
#     # def trap(self, height: List[int]) -> int:
#     #     n = len(height)
#     #     if n < 3: return 0
#     #     LeftMax = [0]*n
#     #     RightMax = [0]*n
#     #     LeftMax[0] = height[0]
#     #     RightMax[n-1] = height[n-1]
#     #     for i in range(1,n):
#     #         LeftMax[i] = max(LeftMax[i-1],height[i])
#     #         RightMax[n-i-1] = max(RightMax[n-i],height[n-i-1])
#     #     res = 0
#     #     for i in range(n):
#     #         res = res + min(LeftMax[i],RightMax[i]) - height[i]
#     #     return res
#     def trap(self, height: List[int]) -> int:
#         ans = 0
#         left, right = 0, len(height) - 1
#         leftMax = rightMax = 0
#         while left < right:
#             leftMax = max(leftMax, height[left])
#             rightMax = max(rightMax, height[right])
#             if height[left] < height[right]:
#                 ans += leftMax - height[left]
#                 left += 1
#             else:
#                 ans += rightMax - height[right]
#                 right -= 1
#         return ans


"""46. 全排列"""     
# class Solution:
#     def permute(self, nums: List[int]) -> List[List[int]]:
#         res = []
#         def dfs(nums, cur_ans):
#             if not nums:
#                 res.append(cur_ans)
#             for i in range(len(nums)):
#                 dfs(nums[:i]+nums[i+1:], cur_ans+[nums[i]])
#         dfs(nums,[])
#         return res

"""48. 旋转图像"""   #找规律的题咯，虽然要求原地改动，但实际上至少得用到一个辅助空间
# class Solution:
#     def rotate(self, matrix: List[List[int]]) -> None:
#         n = len(matrix)
#         # 水平翻转
#         for i in range(n // 2):
#             for j in range(n):
#                 matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
#         # 主对角线翻转
#         for i in range(n):
#             for j in range(i):
#                 matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

# """78. 子集"""
# # class Solution:
# #     def subsets(self, nums: List[int]) -> List[List[int]]:
# #         res = []
# #         def dfs(cur_ans, index=0):
# #             res.append(cur_ans)                 
# #             for i in range(index, len(nums)):
# #                 # cur_ans.append(nums[i])            #python的特性导致这样原地改动cur_ans时会影响之前res.append过的结果，所以不能这样操作。
# #                 dfs(cur_ans+[nums[i]], i+1)          #只能通过传参数的方式
# #                 # cur_ans.pop()
# #         dfs([])
# #         return res
# class Solution:
#     def subsets(self, nums: List[int]) -> List[List[int]]:
#         res = []
#         def dfs(cur_ans, index=0):
#             res.append(cur_ans[:])                 #但是这样深复制就可行！
#             for i in range(index, len(nums)):
#                 cur_ans.append(nums[i])           
#                 dfs(cur_ans, i+1)
#                 cur_ans.pop()
#         dfs([])
#         return res

"""49. 字母异位词分组"""  #defaultdict(factory_function=None) ,在key不存在时，会返回工厂函数的默认值，如int返回0， list返回[]， 也可自行定义函数
# class Solution:          #这种题一看就是哈希表
#     def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
#         import collections
#         mp = collections.defaultdict(list)
#         for st in strs:
#             key = "".join(sorted(st))
#             mp[key].append(st)
#         return list(mp.values())

"""53. 最大子数组和"""  #1.动态规划   2.分治法
# class Solution:
#     def maxSubArray(self, nums: List[int]) -> int:
#         dp = [0]*len(nums)
#         dp[0] = nums[0]
#         for i in range(1,len(nums)):
#             if dp[i-1] < 0:
#                 dp[i] = nums[i]
#             else:
#                 dp[i] = nums[i] + dp[i-1]
#         return max(dp)

"""55. 跳跃游戏"""
# class Solution:
#     def canJump(self, nums: List[int]) -> bool:
#         max_distance = 0
#         for i in range(0,len(nums)):
#             if i > max_distance:
#                 return False
#             max_distance = max(max_distance,i+nums[i])
#         return True
        
"""56. 合并区间"""
# class Solution:
#     def merge(self, intervals: List[List[int]]) -> List[List[int]]:
#         intervals.sort(key=lambda x:x[0])
#         res = [intervals[0]]
#         # for i in range(1,len(intervals)):
#         for i in intervals[1:]:
#             if res[-1][1] >= i[0]:
#                 res[-1][1] = max(res[-1][1],i[1])
#             else:
#                 res.append(i)
#         return res

"""62. 不同路径"""
# class Solution:
#     def uniquePaths(self, m: int, n: int) -> int:
#         dp = [[0]*n for _ in range(m)]
#         for i in range(m):
#             dp[i][n-1] = 1
#         for i in range(m-1,-1,-1):
#             for j in range(n-2,-1,-1):
#                 if i < m-1:
#                     dp[i][j] += dp[i+1][j]
#                 if j < n-1:
#                     dp[i][j] += dp[i][j+1]
#         return dp[0][0]

"""64. 最小路径和"""
# class Solution:
#     def minPathSum(self, grid) -> int:
#         for i in range(len(grid)):
#             for j in range(len(grid[0])):
#                 if i == j == 0: continue
#                 elif i == 0:  grid[i][j] = grid[i][j - 1] + grid[i][j]
#                 elif j == 0:  grid[i][j] = grid[i - 1][j] + grid[i][j]
#                 else: grid[i][j] = min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j]
#         return grid[-1][-1]

"""70. 爬楼梯"""
# class Solution:
#     def climbStairs(self, n: int) -> int:
#         if n<3 : return n
#         a = 1
#         b = 2
#         for i in range(2,n):
#             c = a + b
#             a = b
#             b = c
#         return c

"""72. 编辑距离"""   #dp 
# class Solution:
#     def minDistance(self, word1: str, word2: str) -> int:
#         m , n = len(word1) , len(word2)
#         dp = [[0]*(n+1) for _ in range(m+1)]
#         for i in range(m+1):
#             dp[i][0] = i
#         for j in range(n+1):
#             dp[0][j] = j
#         for i in range(1,m+1):
#             for j in range(1,n+1):
#                 if word1[i-1] == word2[j-1]:
#                     dp[i][j] = dp[i-1][j-1]
#                 else:
#                     dp[i][j] = min(dp[i-1][j-1],dp[i][j-1],dp[i-1][j]) + 1
#         return dp[m][n]

"""75. 颜色分类""" 
# class Solution:
#     def sortColors(self, nums: List[int]) -> None:
#         n = len(nums)
#         index_0, index_2 = 0, n-1
#         i = 0
#         while i < index_2+1:
#             if nums[i] == 2:
#                 nums[i],nums[index_2] = nums[index_2],nums[i]
#                 index_2 -= 1
#             elif nums[i] == 0:
#                 nums[i],nums[index_0] = nums[index_0],nums[i]
#                 index_0 += 1
#                 i += 1
#             else:
#                 i += 1

"""76. 最小覆盖子串""" 
# class Solution:
#     def minWindow(self, s: str, t: str) -> str:
#         m,n = len(s),len(t)
#         import collections
#         need = collections.defaultdict(int)
#         for i in range(n):
#             need[t[i]] += 1
#         def judge(dict):
#             for _,value in dict.items():
#                 if value > 0:
#                     return True   #仍然需要
#             return False
#         st,ed = 0,0
#         min_length,st_record,ed_record = 100000,0,0
#         flag = 1
#         while ed < m:
#             if s[ed] in t and flag:
#                 need[s[ed]] -= 1
#             if judge(need):
#                 ed += 1
#                 flag = 1 
#             else:
#                 flag = 0
#                 if min_length > ed-st+1:
#                     min_length = ed-st+1
#                     st_record,ed_record = st,ed
#                 st += 1
#                 if s[st-1] in t:
#                     need[s[st-1]] += 1
                    
#         if min_length != 100000:
#             return s[st_record:ed_record+1]
#         return ''

"""79. 单词搜索""" 
# class Solution:
#     def exist(self, board: List[List[str]], word: str) -> bool:
#         res = []
#         m,n = len(board), len(board[0])
#         def dfs(word,i,j,cur_ans):
#             if not word:
#                 res.append(1)
#                 return
#             if i>0 and board[i-1][j] == word[0] and (i-1,j) not in cur_ans:
#                 dfs(word[1:],i-1,j,cur_ans+[(i-1,j)])
#             if i<m-1 and board[i+1][j] == word[0] and (i+1,j) not in cur_ans:
#                 dfs(word[1:],i+1,j,cur_ans+[(i+1,j)])
#             if j>0 and board[i][j-1] == word[0] and (i,j-1) not in cur_ans:
#                 dfs(word[1:],i,j-1,cur_ans+[(i,j-1)])
#             if j<n-1 and board[i][j+1] == word[0] and (i,j+1) not in cur_ans:
#                 dfs(word[1:],i,j+1,cur_ans+[(i,j+1)])
#         for i in range(m):
#             for j in range(n):
#                 if board[i][j] == word[0]:
#                     dfs(word[1:],i,j,[(i,j)])
#                     if res:
#                         return True
#         return False

"""84. 柱状图中最大的矩形""" #单调栈：在一维数组中对每一个数找到第一个比自己小的元素。这类“在一维数组中找第一个满足某种条件的数”的场景就是典型的单调栈应用场景。
# class Solution:
#     def largestRectangleArea(self, heights: List[int]) -> int:
#         size = len(heights)
#         res = 0
#         heights = [0] + heights + [0]
#         # 先放入哨兵结点，在循环中就不用做非空判断
#         stack = [0]
#         size += 2

#         for i in range(1, size):
#             while heights[i] < heights[stack[-1]]:
#                 cur_height = heights[stack.pop()]
#                 cur_width = i - stack[-1] - 1
#                 res = max(res, cur_height * cur_width)
#             stack.append(i)
#         return res

"""85. 最大矩形"""   #可以用单调栈优化
# class Solution:
#     def maximalRectangle(self, matrix: List[List[str]]) -> int:
#         m,n = len(matrix),len(matrix[0])
#         dp = [[0]*n for _ in range(m)]
#         for i in range(n):
#             dp[0][i] = int(matrix[0][i])
#         for i in range(1,m):
#             for j in range(n):
#                 if matrix[i][j] == '1':
#                     dp[i][j] = dp[i-1][j] + 1
#                 else:
#                     dp[i][j] = 0
#         max_area = 0
#         for i in range(m):
#             for j in range(n):
#                 if dp[i][j]*n <= max_area:
#                     continue
#                 cnt = 1
#                 for k in range(j+1,n):
#                     if dp[i][k] < dp[i][j]:
#                         break
#                     cnt += 1
#                 for k in range(j-1,-1,-1):
#                     if dp[i][k] < dp[i][j]:
#                         break
#                     cnt += 1
#                 max_area = max(max_area, dp[i][j]*cnt)
#         return max_area

"""94. 二叉树的中序遍历""" 
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# class Solution:
#     def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
#         res = []
#         def inorder(node):
#             if not node:
#                 return
#             inorder(node.left)
#             res.append(node.val)
#             inorder(node.right)
#         inorder(root)
#         return res

"""98. 验证二叉搜索树""" 
# class Solution:
#     def isValidBST(self, root: Optional[TreeNode]) -> bool:
#         res = []
#         def inorder(node):
#             if not node:
#                 return
#             inorder(node.left)
#             res.append(node.val)
#             inorder(node.right)
#         inorder(root)
#         for i in range(len(res)-1):
#             if res[i] >= res[i+1]:
#                 return False
#         return True

"""101. 对称二叉树""" 
# class Solution:
#     def isSymmetric(self, root: Optional[TreeNode]) -> bool:
#         res_left = []
#         res_right = []
#         def left(node):
#             if not node:
#                 res_left.append(None)
#                 return
#             left(node.left)
#             left(node.right)
#             res_left.append(node.val)
#         def right(node):
#             if not node:
#                 res_right.append(None)
#                 return
#             right(node.right)
#             right(node.left)
#             res_right.append(node.val)
#         left(root.left)
#         right(root.right)
#         return res_left == res_right

# class Solution(object):
# 	def isSymmetric(self, root):
# 		def dfs(left,right):
# 			# 递归的终止条件是两个节点都为空
# 			# 或者两个节点中有一个为空
# 			# 或者两个节点的值不相等
# 			if not (left or right):
# 				return True
# 			if not (left and right):
# 				return False
# 			if left.val!=right.val:
# 				return False
# 			return dfs(left.left,right.right) and dfs(left.right,right.left)
# 		return dfs(root.left,root.right)

#还可以用队列来做迭代判断

"""102.二叉树的层序遍历""" 
# class Solution:
#     def levelOrder(self, root: TreeNode) -> List[List[int]]:
#         if not root:
#             return []
#         que = [root]
#         res = []
#         while que:
            # cur_ans = []
            # cur_ans.append(que[0].val)
            # if que[0].left:
            #     que.append(que[0].left)
            # if que[0].right:
            #     que.append(que[0].right)
            # que.pop(0)
            # res.append(cur_ans)
#         return res

"""104. 二叉树的最大深度""" 
# class Solution:
#     def __init__(self) -> None:
#         self.max_depth = 0

#     def maxDepth(self, root: Optional[TreeNode]) -> int:
#         def inorder(node, depth=0):
#             if not node:
#                 self.max_depth = max(self.max_depth, depth)
#                 return
#             depth += 1
#             inorder(node.left, depth)
#             inorder(node.right, depth)
#         inorder(root)
#         return self.max_depth

# class Solution:
#     def maxDepth(self, root):
#         if root is None: 
#             return 0 
#         else: 
#             left_height = self.maxDepth(root.left) 
#             right_height = self.maxDepth(root.right) 
#             return max(left_height, right_height) + 1 

"""105. 从前序与中序遍历序列构造二叉树"""        #递归经典题目，如何构造递归条件
# class Solution:
#     def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
#         inorder_index = {val:index for index,val in enumerate(inorder)}
#         n = len(preorder)
#         def bulid_Tree(pre_start, pre_end, in_start, in_end):
#             if pre_start > pre_end:
#                 return None
#             root = TreeNode(preorder[pre_start])
#             root_index = inorder_index[root.val]
#             num_left_node = root_index - in_start
#             root.left =  bulid_Tree(pre_start+1, pre_start+num_left_node, in_start, root_index-1)
#             root.right = bulid_Tree(pre_start+num_left_node+1, pre_end, root_index+1, in_end)
#             return root
#         return bulid_Tree(0, n-1, 0, n-1)

"""114. 二叉树展开为链表"""
# class Solution:
#     def flatten(self, root: TreeNode) -> None:
#         nodes = []
#         def pre_order(node):
#             if not node:
#                 return
#             nodes.append(node)
#             pre_order(node.left)
#             pre_order(node.right)
#         pre_order(root)
#         for i in range(len(nodes)-1):
#             nodes[i].right = nodes[i+1]
#             nodes[i].left = None

# class Solution:
#     def flatten(self, root: TreeNode) -> None:
#         while root:
#             if not root.left:
#                 root = root.right
#             else:
#                 tmp = root.left
#                 while tmp.right:
#                     tmp = tmp.right
#                 tmp.right = root.right
#                 root.right = root.left
#                 root.left = None
#                 root = root.right

"""121. 买卖股票的最佳时机"""
# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         n = len(prices)
#         max_profit = 0
#         max_price = prices[-1]
#         for i in range(n-2,-1,-1):
#             max_price = max(prices[i], max_price)
#             max_profit = max(max_profit, max_price - prices[i])
#         return max_profit

"""124. 二叉树中的最大路径和"""
# class Solution:
#     def __init__(self) -> None:
#         self.max_sum = -1000

#     def maxPathSum(self, root: Optional[TreeNode]) -> int:
#         def node_gain(node):
#             if not node:
#                 return 0
#             left_gain = max(node_gain(node.left),0)
#             right_gain = max(node_gain(node.right),0)
#             cur_value = node.val + left_gain + right_gain
#             self.max_sum = max(self.max_sum, cur_value)
#             return node.val + max(left_gain, right_gain)
#         node_gain(root)
#         return self.max_sum

"""128. 最长连续序列"""
# class Solution:
#     def longestConsecutive(self, nums: List[int]) -> int:
#         max_length = 0
#         nums_set = set(nums)
#         for num in nums_set:
#             if num - 1 in nums_set:
#                 continue
#             cur_length = 1
#             while num + 1 in nums_set:
#                 cur_length += 1
#                 num += 1
#             max_length = max(max_length, cur_length)
#         return max_length

"""136. 只出现一次的数字"""
from functools import lru_cache, reduce
# class Solution:
#     def singleNumber(self, nums: List[int]) -> int:
#         return reduce(lambda x, y: x ^ y, nums)

"""139. 单词拆分"""
# class Solution:
#     def wordBreak(self, s: str, wordDict: List[str]) -> bool:       
#         n=len(s)
#         dp=[False]*(n+1)
#         dp[0]=True
#         for i in range(n):
#             for j in range(i+1,n+1):
#                 if(dp[i] and (s[i:j] in wordDict)):
#                     dp[j]=True
#         return dp[-1]

# class Solution:
#     def __init__(self) -> None:
#         self.res = False

#     def wordBreak(self, s: str, wordDict: List[str]) -> bool:
#         import functools                  #back_trace回溯可以用这个，不然会超时
#         @functools.lru_cache(None)
#         def dfs(remain_word):
#             if not remain_word:
#                 self.res = True
#                 return
#             for i in range(1, len(remain_word)+1):
#                 if remain_word[:i] in wordDict:
#                     dfs(remain_word[i:])
#         dfs(s)
#         return self.res

"""141. 环形链表"""    #快慢指针！
# class Solution:
#     def hasCycle(self, head: Optional[ListNode]) -> bool:
#         node_set = set()             #用set存储起来，查找时更快
#         while head:
#             if head in node_set:
#                 return True
#             node_set.add(head)
#             head = head.next
#         return False

# class Solution:
#     def hasCycle(self, head: ListNode) -> bool:
#         fast = slow = head
#         while fast and fast.next:
#             fast = fast.next.next
#             slow = slow.next
#             if fast == slow:
#                 return True
#         return False

"""142. 环形链表 II"""
# class Solution:
#     def detectCycle(self, head: ListNode) -> ListNode:
#         node_set = set()             #用set存储起来，查找时更快
#         while head:
#             if head in node_set:
#                 return head
#             node_set.add(head)
#             head = head.next
#         return None

# class Solution(object):
#     def detectCycle(self, head):
#         fast, slow = head, head
#         while True:
#             if not (fast and fast.next): return
#             fast, slow = fast.next.next, slow.next
#             if fast == slow: break
#         fast = head
#         while fast != slow:
#             fast, slow = fast.next, slow.next
#         return fast

"""146. LRU 缓存"""      #ordered dict实际上就是用双链表+哈希表实现的  , 因为从链表中删除一个节点需要访问该节点前驱，所以需要双向链表
# class DLinkedNode:
#     def __init__(self, key=0, value=0):
#         self.key = key
#         self.value = value
#         self.prev = None
#         self.next = None

# class LRUCache:
#     def __init__(self, capacity: int):
#         self.cache = dict()
#         self.head = DLinkedNode()
#         self.tail = DLinkedNode()
#         self.head.next = self.tail
#         self.tail.prev = self.head
#         self.capacity = capacity
#         self.size = 0

#     def get(self, key: int) -> int:
#         if key not in self.cache:
#             return -1
#         node = self.cache[key]
#         self.moveToHead(node)
#         return node.value

#     def put(self, key: int, value: int) -> None:
#         if key not in self.cache:
#             node = DLinkedNode(key, value)
#             self.cache[key] = node
#             self.addToHead(node)
#             self.size += 1
#             if self.size > self.capacity:
#                 removed = self.removeTail()
#                 self.cache.pop(removed.key)
#                 self.size -= 1
#         else:
#             node = self.cache[key]
#             node.value = value
#             self.moveToHead(node)
    
#     def addToHead(self, node):
#         node.prev = self.head
#         node.next = self.head.next
#         self.head.next.prev = node
#         self.head.next = node
    
#     def removeNode(self, node):
#         node.prev.next = node.next
#         node.next.prev = node.prev

#     def moveToHead(self, node):
#         self.removeNode(node)
#         self.addToHead(node)

#     def removeTail(self):
#         node = self.tail.prev
#         self.removeNode(node)
#         return node

"""148. 排序链表"""           #归并排序。    若是采用递归则递归调用的栈空间就使得空间复杂度不为1   #归并排序自顶向下用递归，而自底向上则用迭代
# class Solution:
#     def sortList(self, head: ListNode) -> ListNode:
#         def sortFunc(head: ListNode, tail: ListNode) -> ListNode:
#             if not head:
#                 return head
#             if head.next == tail:
#                 head.next = None
#                 return head
#             slow = fast = head
#             while fast != tail:
#                 slow = slow.next
#                 fast = fast.next
#                 if fast != tail:
#                     fast = fast.next
#             mid = slow
#             return merge(sortFunc(head, mid), sortFunc(mid, tail))
        
#         def merge(head1, head2):
#             if not head1:
#                 return head2
#             if not head2:
#                 return head1
#             if head1.val <= head2.val:
#                 head1.next = merge(head1.next, head2)
#                 return head1
#             else:
#                 head2.next = merge(head2.next, head1)
#                 return head2
#         return sortFunc(head,None)

"""152. 乘积最大子数组"""   #动态规划只需要用到dp[i-1]时，完全不需要定义数组，只需一个变量不断覆盖就行。
# class Solution:
#     def maxProduct(self, nums: List[int]) -> int:
#         max_value = 1
#         min_value = 1
#         res = float(-inf)
#         for i in nums:
#             if i < 0:
#                 max_value, min_value = min_value, max_value
#             max_value = max(max_value*i, i)
#             min_value = min(min_value*i, i)
#             res = max(res, max_value)
#         return res

# class Solution:
#     def maxProduct(self, nums: List[int]) -> int:
#         n = len(nums)
#         if n == 0: return 0
#         dp_max = nums[0]
#         dp_min = nums[0]
#         res = nums[0]
#         for i in nums[1:]:
#             pre_max = dp_max
#             dp_max = max(dp_min*i, max(dp_max*i, i))
#             dp_min = min(dp_min*i, min(pre_max*i, i))
#             res = max(res, dp_max)
#         return res

"""155. 最小栈"""    #辅助栈
# class MinStack:

#     def __init__(self):
#         self.stack = []


#     def push(self, val: int) -> None:
#         if not self.stack:
#             self.stack.append((val,val))
#         else:
#             self.stack.append((val,min(val,self.stack[-1][1])))

#     def pop(self) -> None:
#         self.stack.pop()

#     def top(self) -> int:
#         return self.stack[-1][0]

#     def getMin(self) -> int:
#         return self.stack[-1][1]

"""160. 相交链表""" 
# class Solution:
#     def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
#         temp = set()
#         while headA:
#             temp.add(headA)
#             headA = headA.next
#         while headB:
#             if headB in temp:
#                 return headB
#             headB = headB.next
#         return None

# class Solution:
#     def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
#         p_A = headA
#         p_B = headB
#         while p_A or p_B:
#             if p_A == p_B:
#                 return p_A
#             if p_A:
#                 p_A = p_A.next
#             else:
#                 p_A = headB
#             if p_B:
#                 p_B = p_B.next
#             else:
#                 p_B = headA
#         return None

"""169. 多数元素""" 
# class Solution:
#     def majorityElement(self, nums: List[int]) -> int:
#         def majority_element_rec(lo, hi) -> int:
#             if lo == hi:
#                 return nums[lo]
#             mid = (hi - lo) // 2 + lo
#             left = majority_element_rec(lo, mid)
#             right = majority_element_rec(mid + 1, hi)
#             if left == right:
#                 return left
#             left_count = sum(1 for i in range(lo, hi + 1) if nums[i] == left)
#             right_count = sum(1 for i in range(lo, hi + 1) if nums[i] == right)
#             return left if left_count > right_count else right
#         return majority_element_rec(0, len(nums) - 1)

# class Solution:
#     def majorityElement(self, nums: List[int]) -> int:
#         import collections
#         hash_map = collections.defaultdict(int)
#         max_times = 0
#         res = 0
#         for i in nums:
#             hash_map[i] += 1
#             if hash_map[i] > max_times:
#                 max_times = hash_map[i]
#                 res = i
#         return res

"""198. 打家劫舍""" 
# class Solution:
#     def rob(self, nums: List[int]) -> int:
#         if not nums:
#             return 0
#         size = len(nums)
#         if size == 1:
#             return nums[0]
#         dp = [0] * size
#         dp[0] = nums[0]
#         dp[1] = max(nums[0], nums[1])
#         for i in range(2, size):
#             dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
#         return dp[size - 1]

"""200. 岛屿数量"""    #岛屿类型题目用dfs和bfs  , bfs感觉就是dfs的迭代写法，可以减少一点空间复杂度
# class Solution:
#     def numIslands(self, grid: List[List[str]]) -> int:
#         m, n = len(grid), len(grid[0])
#         def dfs(i,j):
#             grid[i][j] = '0'
#             if i < m-1 and grid[i+1][j] == '1':
#                 dfs(i+1,j)
#             if i > 0 and grid[i-1][j] == '1':
#                 dfs(i-1,j)
#             if j < n-1 and grid[i][j+1] == '1':
#                 dfs(i,j+1)
#             if j > 0 and grid[i][j-1] == '1':
#                 dfs(i,j-1)
#         res = 0
#         for i in range(m):
#             for j in range(n):
#                 if grid[i][j] == '1':
#                     res += 1
#                     dfs(i,j)
#         return res

"""206. 反转链表""" 
# class Solution:
#     def reverseList(self, head: ListNode) -> ListNode:
#         pre = None
#         cur = head
#         while(cur != None):
#             next = cur.next
#             cur.next = pre
#             pre = cur
#             cur = next
#         return pre
            
#     def reverseList(self, head: ListNode) -> ListNode:
#         if head is None or head.next is None:
#             return head
        
#         p = self.reverseList(head.next)
#         head.next.next = head
#         head.next = None

#         return p

"""207. 课程表"""       #拓扑排序:对于图 G 中的任意一条有向边 (u,v)，u 在排列中都出现在 v 的前面。
import collections
# class Solution:         #dfs或者bfs   dfs
#     def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
#         edges = collections.defaultdict(list)
#         visited = [0] * numCourses
#         valid = True
#         for info in prerequisites:
#             edges[info[1]].append(info[0])
#         def dfs(u: int):
#             nonlocal valid
#             visited[u] = 1
#             for v in edges[u]:
#                 if visited[v] == 0:
#                     dfs(v)
#                     if not valid:
#                         return
#                 elif visited[v] == 1:
#                     valid = False
#                     return
#             visited[u] = 2

#         for i in range(numCourses):
#             if valid and not visited[i]:
#                 dfs(i)
#             if not valid:
#                 return False
#         return True
# class Solution:   #bfs
#     def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
#         edges = collections.defaultdict(list)
#         indeg = [0] * numCourses
#         result = list()
#         for info in prerequisites:
#             edges[info[1]].append(info[0])
#             indeg[info[0]] += 1
#         q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])
#         while q:
#             u = q.popleft()
#             result.append(u)
#             for v in edges[u]:
#                 indeg[v] -= 1
#                 if indeg[v] == 0:
#                     q.append(v)
#         if len(result) != numCourses:
#             return False
#         return True

"""208. 实现 Trie (前缀树)"""
# class Trie:
#     def __init__(self):
#         self.children = [None] * 26
#         self.isEnd = False
    
#     def searchPrefix(self, prefix: str) -> "Trie":
#         node = self
#         for ch in prefix:
#             ch = ord(ch) - ord("a")
#             if not node.children[ch]:
#                 return None
#             node = node.children[ch]
#         return node

#     def insert(self, word: str) -> None:
#         node = self
#         for ch in word:
#             ch = ord(ch) - ord("a")
#             if not node.children[ch]:
#                 node.children[ch] = Trie()
#             node = node.children[ch]
#         node.isEnd = True

#     def search(self, word: str) -> bool:
#         node = self.searchPrefix(word)
#         return node is not None and node.isEnd

#     def startsWith(self, prefix: str) -> bool:
#         return self.searchPrefix(prefix) is not None

"""215. 数组中的第K个最大元素"""
import random
#快速排序    time:  avg:nlogn  best:nlogn  badest:n**2               space:logn
# class Solution:
#     def sortArray(self, nums: List[int]) -> List[int]:
#         def partition(arr, low, high):
#             pivot = arr[low]                                        # 选取最左边为pivot
#             left, right = low, high     # 双指针
#             while left < right:
#                 while left<right and arr[right] >= pivot:          # 找到右边第一个<pivot的元素
#                     right -= 1
#                 arr[left] = arr[right]                             # 并将其移动到left处
#                 while left<right and arr[left] <= pivot:           # 找到左边第一个>pivot的元素
#                     left += 1
#                 arr[right] = arr[left]                             # 并将其移动到right处
#             arr[left] = pivot           # pivot放置到中间left=right处
#             return left

#         def quickSort(arr, low, high):
#             if low >= high:             # 递归结束
#                 return  
#             mid = partition(arr, low, high)       # 以mid为分割点【非随机选择pivot】
#             quickSort(arr, low, mid-1)              # 递归对mid两侧元素进行排序
#             quickSort(arr, mid+1, high)
#         quickSort(nums, 0, len(nums)-1)             # 调用快排函数对nums进行排序
#         return nums

#归并排序    time:  avg:nlogn  best:nlogn  badest:nlogn              space:n
# class Solution:
#     def sortArray(self, nums: List[int]) -> List[int]:
#         def mergeSort(arr, low, high):
#             if low >= high:                 # 递归结束标志
#                 return
#             mid = low + (high-low)//2       # 中间位置
#             mergeSort(arr, low, mid)        # 递归对前后两部分进行排序
#             mergeSort(arr, mid+1, high)
#             left, right = low, mid+1        # 将arr一分为二：left指向前半部分（已有序），right指向后半部分（已有序）
#             tmp = []                        # 记录排序结果
#             while left <= mid and right <= high:    # 比较排序，优先添加前后两部分中的较小者
#                 if arr[left] <= arr[right]:         # left指示的元素较小
#                     tmp.append(arr[left])
#                     left += 1
#                 else:                               # right指示的元素较小
#                     tmp.append(arr[right])
#                     right += 1
#             while left <= mid:              # 若左半部分还有剩余，将其直接添加到结果中
#                 tmp.append(arr[left])
#                 left += 1
#             # tmp += arr[left:mid+1]        # 等价于以上三行
#             while right <= high:            # 若右半部分还有剩余，将其直接添加到结果中
#                 tmp.append(arr[right])
#                 right += 1
#             # tmp += arr[right:high+1]      # 等价于以上三行
#             arr[low: high+1] = tmp          # [low, high] 区间完成排序
#         mergeSort(nums, 0, len(nums)-1)     # 调用mergeSort函数完成排序
#         return nums

#堆排序      time:  avg:nlogn  best:nlogn  badest:nlogn         space:1          由于二叉树的节点关系，可用位操作
#堆处理海量数据的topK非常合适，在总体数据规模 n 较大，而维护规模 k 较小时，时间复杂度优化明显。
# class Solution:
#     def sortArray(self, nums: List[int]) -> List[int]:
#         def maxHepify(arr, i, end):     # 大顶堆
#             j = 2*i + 1                 # j为i的左子节点【建堆时下标0表示堆顶】
#             while j <= end:             # 自上而下进行调整
#                 if j+1 <= end and arr[j+1] > arr[j]:    # i的左右子节点分别为j和j+1
#                     j += 1                              # 取两者之间的较大者
                
#                 if arr[i] < arr[j]:             # 若i指示的元素小于其子节点中的较大者
#                     arr[i], arr[j] = arr[j], arr[i]     # 交换i和j的元素，并继续往下判断
#                     i = j                       # 往下走：i调整为其子节点j
#                     j = 2*i + 1                 # j调整为i的左子节点
#                 else:                           # 否则，结束调整
#                     break
#         n = len(nums)
#         # 建堆【大顶堆】
#         for i in range(n//2-1, -1, -1):         # 从第一个非叶子节点n//2-1开始依次往上进行建堆的调整
#             maxHepify(nums, i, n-1)
#         # 排序：依次将堆顶元素（当前最大值）放置到尾部，并调整堆
#         for j in range(n-1, -1, -1):
#             nums[0], nums[j] = nums[j], nums[0]     # 堆顶元素（当前最大值）放置到尾部j
#             maxHepify(nums, 0, j-1)                 # j-1变成尾部，并从堆顶0开始调整堆
#         return nums


"""221. 最大正方形"""    #记得回顾84、85中单调栈的做法
# class Solution:
#     def maximalSquare(self, matrix: List[List[str]]) -> int:
#         m, n = len(matrix), len(matrix[0])
#         dp = [[0]*n for _ in range(m)]
#         max_area = 0
#         for j in range(n):
#             dp[0][j] = 1 if matrix[0][j]=='1' else 0
#             max_area = max(max_area, dp[0][j])
#         for i in range(m):
#             dp[i][0] = 1 if matrix[i][0]=='1' else 0
#             max_area = max(max_area, dp[i][0])
#         for i in range(1,m):
#             for j in range(1,n):
#                 if matrix[i][j] == '0':
#                     dp[i][j] = 0
#                 else:
#                     dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
#                     max_area = max(max_area, dp[i][j])
#         return max_area**2

"""226. 翻转二叉树"""
# class Solution:
#     def invertTree(self, root: TreeNode) -> TreeNode:
#         if not root:
#             return root
#         left = self.invertTree(root.left)
#         right = self.invertTree(root.right)
#         root.left, root.right = right, left
#         return root

# class Solution:
#     def invertTree(self, root: TreeNode) -> TreeNode:
#         def back(node):
#             if not node:
#                 return
#             back(node.left)
#             back(node.right)
#             node.left, node.right = node.right, node.left
#         back(root)
#         return root

"""234. 回文链表"""
# class Solution:
#     def isPalindrome(self, head: ListNode) -> bool:
#         if head is None:
#             return True
#         first_half_end = self.end_of_first_half(head)
#         second_half_start = self.reverse_list(first_half_end.next)
#         result = True
#         first_position = head
#         second_position = second_half_start
#         while result and second_position is not None:
#             if first_position.val != second_position.val:
#                 result = False
#             first_position = first_position.next
#             second_position = second_position.next
#         # first_half_end.next = self.reverse_list(second_half_start) 似乎没有完全复原，链表反转后应该与前半部分断开了
#         return result    

#     def end_of_first_half(self, head):
#         fast = head
#         slow = head
#         while fast.next is not None and fast.next.next is not None:
#             fast = fast.next.next
#             slow = slow.next
#         return slow

#     def reverse_list(self, head):
#         previous = None
#         current = head
#         while current is not None:
#             next_node = current.next
#             current.next = previous
#             previous = current
#             current = next_node
#         return previous

"""236. 二叉树的最近公共祖先"""
# class Solution:
#     def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
#         if not root or root==p or root==q:
#             return root
#         left = self.lowestCommonAncestor(root.left, p, q)
#         right = self.lowestCommonAncestor(root.right, p, q)
#         if not left and not right:
#             return None
#         if not left:
#             return right
#         if not right:
#             return left
#         return root

# """238. 除自身以外数组的乘积"""
# class Solution:
#     def productExceptSelf(self, nums: List[int]) -> List[int]:
#         n=len(nums)
#         nums1 = [1]
#         nums2 = [1]
#         for i in nums:
#             nums1.append(i*nums1[-1])
#         for i in nums[-1::-1]:
#             nums2.append(i*nums2[-1])
#         for i in range(n):
#             nums[i] = nums1[i]*nums2[n-i-1]
#         return nums

"""239. 滑动窗口最大值"""     #维护最大值非常合适的数据结构是优先队列（python中的小根堆）
# def heapify(l):     #实现对输入list建小根堆
#     n = len(l)
#     for i in range(n//2-1, -1, -1):
#         min_heapify(l, i, n-1)

# def heappush(l, input):     #向已经是小根堆的list中添加数据并构成新的小根堆
#     l.append(input)
#     n = len(l)
#     j = n//2 - 1
#     while j >= 0:
#         if l[j] > l[n-1]:
#             l[j], l[n-1] = l[n-1], l[j]
#             n = j + 1
#             j = (j+1)//2 - 1
#         else:
#             break

# def heappop(l):    #从小根堆中弹出最小的数据并构成新的小根堆
#     l[0],l[-1] = l[-1],l[0]
#     l.pop()
#     min_heapify(l, 0, len(l)-1)

# def min_heapify(l, start, end):
#     j = 2*start + 1
#     while j <= end:
#         if j+1 <= end and l[j+1] < l[j]:
#             j += 1
#         if l[start] > l[j]:
#             l[start],l[j] = l[j],l[start]
#             start = j
#             j = 2*start +1
#         else:
#             break

import heapq
# class Solution:
#     def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:     #堆方法
#         n = len(nums)
#         # 注意 Python 默认的优先队列是小根堆
#         q = [(-nums[i], i) for i in range(k)]
#         heapq.heapify(q)

#         ans = [-q[0][0]]
#         for i in range(k, n):
#             heapq.heappush(q, (-nums[i], i))
#             while q[0][1] <= i - k:
#                 heapq.heappop(q)
#             ans.append(-q[0][0])
#         return ans

# class Solution:
#     def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:     #双端队列法, deque
#         n = len(nums)
#         q = collections.deque()
#         for i in range(k):
#             while q and nums[i] >= nums[q[-1]]:
#                 q.pop()
#             q.append(i)

#         ans = [nums[q[0]]]
#         for i in range(k, n):
#             while q and nums[i] >= nums[q[-1]]:
#                 q.pop()
#             q.append(i)
#             if q[0] <= i - k:
#                 q.popleft()
#             ans.append(nums[q[0]])
#         return ans

"""240. 搜索二维矩阵 II"""   #查找可用二分查找，但此题利用题干可有更简便方法
# class Solution:
#     def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
#         m, n = len(matrix), len(matrix[0])
#         x, y = 0, n - 1
#         while x < m and y >= 0:
#             if matrix[x][y] == target:
#                 return True
#             if matrix[x][y] > target:
#                 y -= 1
#             else:
#                 x += 1
#         return False

"""279. 完全平方数"""
# class Solution:
#     def numSquares(self, n: int) -> int:
#         dp = [0]*(n+1)
#         for i in range(1,n+1):
#             minn = 10000
#             j = 1
#             while j*j <= i:
#                 minn = min(minn, dp[i - j*j])
#                 j += 1
#             dp[i] = minn + 1
#         return dp[n]

"""283. 移动零"""          #双指针的思想很重要，也并非总是一头一尾，妙啊
# class Solution:
#     def moveZeroes(self, nums: List[int]) -> None:
#         n = len(nums)
#         left = right = 0
#         while right < n:
#             if nums[right] != 0:
#                 nums[left], nums[right] = nums[right], nums[left]
#                 left += 1
#             right += 1

"""287. 寻找重复数"""     #借助环形链表，确实难想
# class Solution:
#     def findDuplicate(self, nums: List[int]) -> int:
#         fast, slow = nums[nums[0]], nums[0]
#         while fast != slow:
#             slow = nums[slow]
#             fast = nums[nums[fast]]
#         fast = 0
#         while fast != slow:
#             fast, slow = nums[fast], nums[slow]
#         return fast

"""297. 二叉树的序列化与反序列化""" #pop(0)可以用deque的popleft替代
# class Codec:
#     def serialize(self, root):
#         if not root:
#             return ""
#         queue = collections.deque([root])
#         res = []
#         while queue:
#             node = queue.popleft()
#             if node:
#                 res.append(str(node.val))
#                 queue.append(node.left)
#                 queue.append(node.right)
#             else:
#                 res.append('None')
#         return '[' + ','.join(res) + ']'

#     def deserialize(self, data):
#         if not data:
#             return []
#         dataList = data[1:-1].split(',')
#         root = TreeNode(int(dataList[0]))
#         queue = collections.deque([root])
#         i = 1
#         while queue:
#             node = queue.popleft()
#             if dataList[i] != 'None':
#                 node.left = TreeNode(int(dataList[i]))
#                 queue.append(node.left)
#             i += 1
#             if dataList[i] != 'None':
#                 node.right = TreeNode(int(dataList[i]))
#                 queue.append(node.right)
#             i += 1
#         return root


"""300. 最长递增子序列"""         #能够用二分的前提在于单调性，一次二分是logn
# class Solution:
#     def lengthOfLIS(self, nums: List[int]) -> int:
#         if not nums:
#             return 0
#         dp = []
#         for i in range(len(nums)):
#             dp.append(1)
#             for j in range(i):
#                 if nums[i] > nums[j]:
#                     dp[i] = max(dp[i], dp[j] + 1)
#         return max(dp)

# class Solution:
#     def lengthOfLIS(self, nums: List[int]) -> int:
#         size = len(nums)
#         if size<2:
#             return size
        
#         cell = [nums[0]]
#         for num in nums[1:]:
#             if num>cell[-1]:
#                 cell.append(num)
#                 continue
            
#             l,r = 0,len(cell)-1
#             while l<r:
#                 mid = l + (r - l) // 2
#                 if cell[mid]<num:
#                     l = mid + 1
#                 else:
#                     r = mid
#             cell[l] = num
#         return len(cell)

"""301. 删除无效的括号"""  #括号有效规则要牢记
# class Solution:     #dfs
#     def removeInvalidParentheses(self, s: str) -> List[str]:
#         res = []
#         lremove, rremove = 0, 0
#         for c in s:
#             if c == '(':
#                 lremove += 1
#             elif c == ')':
#                 if lremove == 0:
#                     rremove += 1
#                 else:
#                     lremove -= 1

#         def isValid(str):
#             cnt = 0
#             for c in str:
#                 if c == '(':
#                     cnt += 1
#                 elif c == ')':
#                     cnt -= 1
#                     if cnt < 0:
#                         return False
#             return cnt == 0

#         def helper(s, start, lremove, rremove):
#             if lremove == 0 and rremove == 0:
#                 if isValid(s):
#                     res.append(s)
#                 return

#             for  i in range(start, len(s)):
#                 if i > start and s[i] == s[i - 1]:
#                     continue
#                 # 如果剩余的字符无法满足去掉的数量要求，直接返回
#                 if lremove + rremove > len(s) - i:
#                     break
#                 # 尝试去掉一个左括号
#                 if lremove > 0 and s[i] == '(':
#                     helper(s[:i] + s[i + 1:], i, lremove - 1, rremove);
#                 # 尝试去掉一个右括号
#                 if rremove > 0 and s[i] == ')':
#                     helper(s[:i] + s[i + 1:], i, lremove, rremove - 1);
#                 # 统计当前字符串中已有的括号数量

#         helper(s, 0, lremove, rremove)
#         return res

# class Solution:   #bfs
#     def removeInvalidParentheses(self, s: str) -> List[str]:
#         def isValid(s):
#             count = 0
#             for c in s:
#                 if c == '(':
#                     count += 1
#                 elif c == ')':
#                     count -= 1
#                     if count < 0:
#                         return False
#             return count == 0
#         ans = []
#         currSet = set([s])
#         while True:
#             for ss in currSet:
#                 if isValid(ss):
#                     ans.append(ss)
#             if len(ans) > 0:
#                 return ans
#             nextSet = set()
#             for ss in currSet:
#                 for i in range(len(ss)):
#                     if i > 0 and ss[i] == s[i - 1]:
#                         continue
#                     if ss[i] == '(' or ss[i] == ')':
#                         nextSet.add(ss[:i] + ss[i + 1:])
#             currSet = nextSet

"""309. 最佳买卖股票时机含冷冻期"""#一种常用的方法是将「买入」和「卖出」分开进行考虑：「买入」为负收益，而「卖出」为正收益。
                                  #dp也可以设置多个状态
# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         n = len(prices)
#         dp = [[0]*3 for _ in range(n)]
#         dp[0][0] = -prices[0]
#         for i in range(1,n):
#             dp[i][0] = max(dp[i-1][0], dp[i-1][2]-prices[i])
#             dp[i][1] = dp[i-1][0] + prices[i]
#             dp[i][2] = max(dp[i-1][1], dp[i-1][2])
#         return max(dp[n-1][1], dp[n-1][2])

"""312. 戳气球"""
# class Solution:
#     def maxCoins(self, nums: List[int]) -> int:
#         #nums首尾添加1，方便处理边界情况
#         nums.insert(0,1)
#         nums.insert(len(nums),1)
#         store = [[0]*(len(nums)) for _ in range(len(nums))]
#         def range_best(i,j):
#             m = 0 
#             #k是(i,j)区间内最后一个被戳的气球
#             for k in range(i+1,j): #k取值在(i,j)开区间中
#                 #以下都是开区间(i,k), (k,j)
#                 left = store[i][k]
#                 right = store[k][j]
#                 m = max(left + nums[i]*nums[k]*nums[j] + right, m)
#             store[i][j] = m
#         #对每一个区间长度进行循环
#         for n in range(2,len(nums)): #区间长度 #长度从3开始，n从2开始
#             #开区间长度会从3一直到len(nums)
#             #因为这里取的是range，所以最后一个数字是len(nums)-1
#             #对于每一个区间长度，循环区间开头的i
#             for i in range(0,len(nums)-n): #i+n = len(nums)-1
#                 #计算这个区间的最多金币
#                 range_best(i,i+n)
#         return store[0][len(nums)-1]

"""322. 零钱兑换""" # ->279
import functools         #记忆化搜索，在python中可以利用@functools.lru_cache()来减少递归的计算量。
# class Solution:
#     def coinChange(self, coins: List[int], amount: int) -> int:
        # dp = [float('inf')] * (amount + 1)
        # dp[0] = 0
        
        # for coin in coins:
        #     for x in range(coin, amount + 1):
        #         dp[x] = min(dp[x], dp[x - coin] + 1)
        # return dp[amount] if dp[amount] != float('inf') else -1 

"""337. 打家劫舍 III"""  #帅！掌握了这种定义多个状态的dp
# class Solution:
#     def rob(self, root: TreeNode) -> int:
#         def houxu(node):
#             if not node:
#                 return [0,0]
#             left = houxu(node.left)
#             right = houxu(node.right)
#             return [left[1]+right[1]+node.val, max(left)+max(right)]
#         return max(houxu(root))
            
"""338. 比特位计数""" # x = x & (x−1) 可将x的二进制表示的最后一个1变为0  , 位运算很快
# class Solution:     
#     def countBits(self, n: int) -> List[int]:
#         highBit = 0
#         bits = [0]*(n+1)
#         for i in range(1,n+1):
#             if i & (i-1) == 0:
#                 highBit = i
#             bits[i] = bits[i-highBit] + 1
#         return bits

# class Solution:  # >>,<<  二进制下的移位操作，⌊x​/2⌋可以由x>>1得到， x除以2的余数可以由x&1得到
#     def countBits(self, n: int) -> List[int]:
#         bits = [0]
#         for i in range(1, n + 1):
#             bits.append(bits[i >> 1] + (i & 1))
#         return bits

"""347. 前 K 个高频元素"""      #堆处理海量数据的topK非常合适，在总体数据规模 n 较大，而维护规模 k 较小时，时间复杂度优化明显。
# class Solution:  #nlogn,要求nlogk则需要用堆
#     def topKFrequent(self, nums: List[int], k: int) -> List[int]:
#         hash_map = collections.defaultdict(int)
#         for i in nums:
#             hash_map[i] += 1
#         temp = []
#         for key, value in hash_map.items():
#             temp.append([value,key])
#         temp.sort(key=lambda x:x[0], reverse=True)
#         return [x[1] for x in temp[:k]]

# class Solution:
#     def topKFrequent(self, nums: List[int], k: int) -> List[int]:
#         def sift_down(arr, root, k):
#             """下沉log(k),如果新的根节点>子节点就一直下沉"""
#             val = arr[root] # 用类似插入排序的赋值交换
#             while root<<1 < k:
#                 child = root << 1
#                 # 选取左右孩子中小的与父节点交换
#                 if child|1 < k and arr[child|1][1] < arr[child][1]:
#                     child |= 1
#                 # 如果子节点<新节点,交换,如果已经有序break
#                 if arr[child][1] < val[1]:
#                     arr[root] = arr[child]
#                     root = child
#                 else:
#                     break
#             arr[root] = val

#         def sift_up(arr, child):
#             """上浮log(k),如果新加入的节点<父节点就一直上浮"""
#             val = arr[child]
#             while child>>1 > 0 and val[1] < arr[child>>1][1]:
#                 arr[child] = arr[child>>1]
#                 child >>= 1
#             arr[child] = val

#         stat = collections.Counter(nums)
#         stat = list(stat.items())
#         heap = [(0,0)]

#         # 构建规模为k+1的堆,新元素加入堆尾,上浮
#         for i in range(k):
#             heap.append(stat[i])
#             sift_up(heap, len(heap)-1) 
#         # 维护规模为k+1的堆,如果新元素大于堆顶,入堆,并下沉
#         for i in range(k, len(stat)):
#             if stat[i][1] > heap[1][1]:
#                 heap[1] = stat[i]
#                 sift_down(heap, 1, k+1) 
#         return [item[0] for item in heap[1:]]

"""394. 字符串解码""" #数字存放在数字栈，字符串存放在字符串栈，遇到右括号时候弹出一个数字栈，字母栈弹到左括号为止
# class Solution:
#     def decodeString(self, s: str) -> str:
#         stack, res, multi = [], "", 0
#         for c in s:
#             if c == '[':
#                 stack.append([multi, res])
#                 res, multi = "", 0
#             elif c == ']':
#                 cur_multi, last_res = stack.pop()
#                 res = last_res + cur_multi * res
#             elif '0' <= c <= '9':
#                 multi = multi * 10 + int(c)            
#             else:
#                 res += c
#         return res

# class Solution:
#     def decodeString(self, s: str) -> str:
#         def dfs(s, i):
#             res, multi = "", 0
#             while i < len(s):
#                 if '0' <= s[i] <= '9':
#                     multi = multi * 10 + int(s[i])
#                 elif s[i] == '[':
#                     i, tmp = dfs(s, i + 1)
#                     res += multi * tmp
#                     multi = 0
#                 elif s[i] == ']':
#                     return i, res
#                 else:
#                     res += s[i]
#                 i += 1
#             return res
#         return dfs(s,0)

"""399. 除法求值"""     #并查集：记录了节点之间的连通关系
class UnionFind:        #带权重的并查集
    def __init__(self):
        """
        记录每个节点的父节点
        记录每个节点到根节点的权重
        """
        self.father = {}
        self.value = {}
    
    def find(self,x):
        """
        查找根节点
        路径压缩
        更新权重
        """
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
        """
        合并两个节点
        """
        root_x,root_y = self.find(x),self.find(y)
        
        if root_x != root_y:
            self.father[root_x] = root_y
            ##### 四边形法则更新根节点的权重
            self.value[root_x] = self.value[y] * val / self.value[x]

    def is_connected(self,x,y):
        """
        两节点是否相连
        """
        return x in self.value and y in self.value and self.find(x) == self.find(y)
    
    def add(self,x):
        """
        添加新节点，初始化权重为1.0
        """
        if x not in self.father:
            self.father[x] = None
            self.value[x] = 1.0


# class Solution:
#     def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
#         uf = UnionFind()
#         for (a,b),val in zip(equations,values):
#             uf.add(a)
#             uf.add(b)
#             uf.merge(a,b,val)
    
#         res = [-1.0] * len(queries)

#         for i,(a,b) in enumerate(queries):
#             if uf.is_connected(a,b):
#                 res[i] = uf.value[a] / uf.value[b]
#         return res

"""406. 根据身高重建队列""" #一般这种数对，还涉及排序的，根据第一个元素正向排序，根据第二个元素反向排序，
                            #或者根据第一个元素反向排序，根据第二个元素正向排序，往往能够简化解题过程。
# class Solution:             #先排序，再插队
#     def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
#         res = []
#         people = sorted(people, key = lambda x: (-x[0], x[1]))
#         for p in people:
#             if len(res) <= p[1]:
#                 res.append(p)
#             elif len(res) > p[1]:
#                 res.insert(p[1], p)
#         return res

"""416. 分割等和子集"""
# class Solution:
#     def canPartition(self, nums: List[int]) -> bool:
#         n = len(nums)
#         if n < 2:
#             return False
        
#         total = sum(nums)
#         maxNum = max(nums)
#         if total & 1:  #算奇偶的一种特别酷的方法
#             return False
        
#         target = total // 2
#         if maxNum > target:
#             return False
        
#         dp = [[False] * (target + 1) for _ in range(n)]
#         for i in range(n):
#             dp[i][0] = True
        
#         dp[0][nums[0]] = True
#         for i in range(1, n):
#             num = nums[i]
#             for j in range(1, target + 1):
#                 if j >= num:
#                     dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num]       #因为每次只使用了上一行的状态，可以考虑空间优化
#                 else:
#                     dp[i][j] = dp[i - 1][j]
        
#         return dp[n - 1][target]

"""437. 路径总和 III"""
# class Solution:
#     def pathSum(self, root: TreeNode, targetSum: int) -> int:
#         def rootSum(root, targetSum):
#             if root is None:
#                 return 0

#             ret = 0
#             if root.val == targetSum:
#                 ret += 1

#             ret += rootSum(root.left, targetSum - root.val)
#             ret += rootSum(root.right, targetSum - root.val)
#             return ret
        
#         if root is None:
#             return 0
            
#         ret = rootSum(root, targetSum)
#         ret += self.pathSum(root.left, targetSum)
#         ret += self.pathSum(root.right, targetSum)
#         return ret

# class Solution:                 #前缀和:达到当前元素的路径上，之前所有元素的和
#     def pathSum(self, root: TreeNode, targetSum: int) -> int:
#         prefix = collections.defaultdict(int)
#         prefix[0] = 1       #为满足自身节点值就等于targetSum的节点提供路径
#         def dfs(root, cur):
#             if not root:
#                 return 0
#             res = 0
#             cur += root.val
#             res += prefix[cur - targetSum]
#             prefix[cur] += 1
#             res += dfs(root.left, cur)
#             res += dfs(root.right, cur)
#             prefix[cur] -= 1
#             return res
#         return dfs(root, 0)

"""438.找到字符串中所有字母异位词"""      #ord('a')=97
# class Solution:
#     def findAnagrams(self, s: str, p: str) -> List[int]:
#         s_len, p_len = len(s), len(p)
#         if s_len < p_len:
#             return []
#         ans = []
#         s_count = [0] * 26
#         p_count = [0] * 26
#         for i in range(p_len):
#             s_count[ord(s[i]) - 97] += 1
#             p_count[ord(p[i]) - 97] += 1
#         if s_count == p_count:
#             ans.append(0)
#         for i in range(s_len - p_len):
#             s_count[ord(s[i]) - 97] -= 1
#             s_count[ord(s[i + p_len]) - 97] += 1
#             if s_count == p_count:
#                 ans.append(i + 1)
#         return ans

"""448. 找到所有数组中消失的数字"""  #不使用额外空间则原地修改或者位操作
# class Solution:
#     def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
#         # n = len(nums)
#         # for num in nums:
#         #     nums[(num-1)%n] += n
#         #return [i+1 for i,num in enumerate(nums) if num <= n]
#         for n in nums:   #和上面的解法一个道理
#             nums[abs(n)-1] = - abs(nums[abs(n)-1])
#         return [i+1 for i,num in enumerate(nums) if num > 0]

"""461. 汉明距离"""  
# class Solution:
#     def hammingDistance(self, x: int, y: int) -> int:
#         # res = 0
#         # while x or y:
#         #     if x & 1 != y & 1:
#         #         res += 1
#         #     x >>= 1
#         #     y >>= 1
#         # return res
#         res = 0
#         s = x ^ y
#         while s:
#             res += s & 1
#             s >>= 1
#         return res

"""494. 目标和"""  
# class Solution:   #python dfs超时
#     def findTargetSumWays(self, nums: List[int], target: int) -> int:
#         def dfs(target, idx, total):
#             res = 0
#             if idx == len(nums) - 1:
#                 if target == total or target == total - 2*nums[idx]:
#                     res += 1
#                 return res
#             res += dfs(target, idx+1, total)
#             res += dfs(target, idx+1, total - 2*nums[idx])
#             return res
#         return dfs(target, 0, sum(nums))

# class Solution:   #转换为背包问题使用动规，核心思想在于符号为负的元素和可以用sum和target表示出来，即可转化为背包问题
#                   # dp[i][j]表示在数组nums的前i个数中选取元素，使得这些元素之和等于j的方案数
#     def findTargetSumWays(self, nums: List[int], target: int) -> int:
#         temp = sum(nums) - target
#         if temp < 0 or temp & 1:
#             return 0
#         neg =  int(temp / 2)
#         n = len(nums)
#         dp = [[0]*(neg+1) for _ in range(n+1)]
#         for j in range(neg+1):
#             dp[0][j] = 1 if j == 0 else 0
#         for i in range(1,n+1):
#             for j in range(neg+1):
#                 if j < nums[i-1]:
#                     dp[i][j] = dp[i-1][j]
#                 else:
#                     dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
#         return dp[n][neg]

"""538. 把二叉搜索树转换为累加树"""  
# class Solution:
#     def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
#         val = 0
#         def mid_order(node):
#             nonlocal val            #这一点还挺关键，如果把这个val当作函数参数反而不好处理
#             if not node:
#                 return 
#             mid_order(node.right)
#             val += node.val
#             node.val = val
#             mid_order(node.left)
#         mid_order(root)
#         return root

"""543. 二叉树的直径"""  
# class Solution:
#     def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
#         max_length = 0
#         def post_order(node):
#             if not node:
#                 return -1, -1
#             nonlocal max_length
#             left_length, right_length = 0, 0
#             left_length = max(post_order(node.left)) + 1
#             right_length = max(post_order(node.right)) + 1
#             max_length = max(max_length, left_length+right_length)
#             return left_length, right_length
#         post_order(root)
#         return max_length

"""560. 和为 K 的子数组"""  #前缀和 + 哈希表      前缀和一般都会有哨兵
# class Solution:
#     def subarraySum(self, nums: List[int], k: int) -> int:
#         hash_map = collections.defaultdict(int)
#         hash_map[0] += 1
#         pre = 0
#         res = 0
#         for num in nums:
#             pre += num
#             res += hash_map[pre - k]
#             hash_map[pre] += 1
#         return res

'''581. 最短无序连续子数组''' #把这个数组分成三段,左段和右段是标准的升序数组,中段数组虽是无序的,但满足最小值大于左段的最大值,最大值小于右段的最小值。
# class Solution:            #找中段的左右边界
#     def findUnsortedSubarray(self, nums: List[int]) -> int:
#         n = len(nums)
#         maxn, right = float("-inf"),-1
#         minn, left = float("inf"),-1
#         for i in range(n):
#             if maxn <= nums[i]:
#                 maxn = nums[i]
#             else:           #进入到右段后则最大值不断刷新，而指针不变
#                 right = i
#             if minn >= nums[n-i-1]:
#                 minn = nums[n-i-1]
#             else:          #进入到左段后则最小值不断刷新，而指针不变
#                 left = n - i - 1
#         return 0 if left == -1 else right - left + 1

"""617. 合并二叉树"""
# class Solution:
#     def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
#         if not root1:
#             return root2
#         if not root2:
#             return root1
#         new_node = TreeNode(root1.val+root2.val)
#         new_node.left = self.mergeTrees(root1.left, root2.left)
#         new_node.right = self.mergeTrees(root1.right, root2.right)
#         return new_node

"""621. 任务调度器"""  #桶思想
# class Solution:
#     def leastInterval(self, tasks: List[str], n: int) -> int:
#         freq = collections.Counter(tasks)
#         # 最多的执行次数
#         maxExec = max(freq.values())
#         # 具有最多执行次数的任务数量
#         maxCount = sum(1 for v in freq.values() if v == maxExec)
#         return max((maxExec - 1) * (n + 1) + maxCount, len(tasks))


"""647. 回文子串"""
# class Solution:
#     def countSubstrings(self, s: str) -> int:
#         n = len(s)
#         dp = [[0]*n for _ in range(n)]
#         nums = 0
#         for i in range(n):
#             dp[i][i] = 1
#             nums += 1
#         for i in range(n-1):
#             if s[i] == s[i+1]:
#                 dp[i][i+1] = 1
#                 nums += 1
#         for length in range(3,n+1):
#             for i in range(0,n-1):
#                 j = length + i -1
#                 if j >= n:
#                     continue
#                 if s[i] == s[j]:
#                     dp[i][j] = dp[i+1][j-1]
#                     if dp[i][j] == 1:
#                         nums += 1
#         return nums


"""739. 每日温度""" #典型的单调栈
# class Solution:
#     def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
#         n = len(temperatures)
#         ans = [0]*n
#         stack = []
#         for i in range(n):
#             while stack and temperatures[i] > temperatures[stack[-1]]:
#                 ans[stack[-1]] = i - stack[-1]
#                 stack.pop()
#             stack.append(i)
#         return ans
