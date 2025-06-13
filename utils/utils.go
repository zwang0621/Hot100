package utils

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
	"strconv"
)

/*
1.两数之和
*/
func TwoSum(nums []int, target int) []int { //双层for
	for i := 0; i < len(nums); i++ {
		for j := i + 1; j < len(nums); j++ {
			if nums[i]+nums[j] == target {
				return []int{i, j}
			}
		}
	}
	return nil
}

func TwoSum2(nums []int, target int) []int { //哈希表
	hushmap := make(map[int]int)
	for i, x := range nums {
		if v, ok := hushmap[target-x]; ok {
			return []int{v, i}
		}
		hushmap[x] = i
	}
	return nil
}

/*
49.字母异位词分组
*/
func GroupAnagrams(strs []string) [][]string {
	//1.先创建一个map，记录下排序后的key和它对应的所有切片元素
	strs_map := make(map[string][]string) //键是字符串，值是切片
	for _, str := range strs {
		str_byte := []byte(str)
		sort.Slice(str_byte, func(i, j int) bool {
			return str_byte[i] < str_byte[j]
		})
		str_new := string(str_byte)
		strs_map[str_new] = append(strs_map[str_new], str)
	}

	//2.遍历之前的map，用一个二维切片返回结果
	s := make([][]string, 0, len(strs_map)) //重要！！必须填满三个参数，长度容量都需要，否则返回的s会有空的切片
	for _, v := range strs_map {
		s = append(s, v)
	}
	return s
}

/*
128.最长连续序列
*/
func LongestConsecutive1(nums []int) int { //时间复杂度 O(nlogn) 排序消耗
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return 1
	}
	sort.Ints(nums)
	start := nums[0]
	count := 0
	max_count := count
	for i := 1; i < len(nums); i++ {
		if nums[i] == start+1 {
			count += 1
			start = nums[i]
		} else if nums[i] == start {
			continue
		} else {
			start = nums[i]
			if max_count < count {
				max_count = count
			}
			count = 0
		}
	}
	if max_count < count {
		max_count = count
	}
	return max_count + 1
}

func LongestConsecutive2(nums []int) int { //哈希表
	numset := make(map[int]bool)
	for i := 0; i < len(nums); i++ {
		numset[nums[i]] = true
	}

	longest := 0
	//for-range循环不一定非得要key-value同时遍历，可以只遍历键，或者只遍历值都可以
	for num := range numset { //要注意这里必须对numset进行循环，因为nums中可能会有多个重复数字导致算法超时
		if !numset[num-1] {
			current_num := num
			current_len := 1

			for numset[current_num+1] {
				current_num += 1
				current_len += 1
			}

			if current_len > longest {
				longest = current_len
			}
		}
	}
	return longest
}

/*
283.移动0
*/
func MoveZeroes(nums []int) { //双指针
	left, right, length := 0, 0, len(nums)
	for right < length {
		if nums[right] != 0 {
			nums[left], nums[right] = nums[right], nums[left]
			left += 1
		}
		right += 1
	}
}

/*
11.盛最多水的容器
*/
func MaxArea(height []int) int { //双指针   Set one pointer to the left and one to the right of the array. Always move the pointer that points to the lower line.
	left, right := 0, len(height)-1
	area := 0
	max_area := area
	for left != right {
		if height[left] > height[right] {
			area = height[right] * (right - left)
		} else {
			area = height[left] * (right - left)
		}
		if area > max_area {
			max_area = area
		}
		if height[left] > height[right] {
			right--
		} else {
			left++
		}
	}
	return max_area
}

/*
15. 三数之和
*/
func ThreeSum(nums []int) [][]int { //双指针
	sort.Ints(nums)
	if nums[0] > 0 {
		return nil
	}
	fmt.Println(nums)
	result := make([][]int, 0)
	for i := 0; i < len(nums)-2; i++ {
		//避免固定值重复
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		fixed_num := nums[i]
		j, k := i+1, len(nums)-1 //左右指针，分别指向当前最小和最大的数
		for j < k {
			if fixed_num+nums[j]+nums[k] == 0 {
				result = append(result, []int{fixed_num, nums[j], nums[k]})
				j++
				k--
				for j < k && nums[j] == nums[j-1] {
					j++
				}
				for j < k && nums[k] == nums[k+1] {
					k--
				}
			} else if fixed_num+nums[j]+nums[k] < 0 {
				j++
			} else {
				k--
			}
		}
	}
	return result
}

/*
3.无重复字符的最长字串
*/
func LengthOfLongestSubstring(s string) int { //滑动窗口
	max_length := 0
	found_str := make(map[byte]int)
	start := 0
	for end := 0; end < len(s); end++ {
		if index, found := found_str[s[end]]; found && index >= start {
			start = index + 1
		}
		found_str[s[end]] = end
		if max_length < end-start+1 {
			max_length = end - start + 1
		}
	}
	return max_length
}

/*
438.找到字符串中的所有字母异位词
*/
func FindAnagrams(s string, p string) []int { //滑动窗口
	length_s := len(s)
	length_p := len(p)
	if length_s < length_p {
		return nil
	}
	s_count := [26]int{}  //记录当前窗口中s中字母出现次数的数组
	p_count := [26]int{}  //记录p中字母出现次数的数组
	res := make([]int, 0) //结果列表
	for i := 0; i < len(p); i++ {
		s_count[s[i]-'a']++
		p_count[p[i]-'a']++
	}
	if s_count == p_count {
		res = append(res, 0)
	}
	for j := length_p; j < len(s); j++ {
		s_count[s[j]-'a']++
		s_count[s[j-length_p]-'a']--
		if s_count == p_count {
			res = append(res, j-length_p+1) //当每次到了滑动窗口的最后一个字母时，这时才知道是否和p_count相等
		}
	}
	return res
}

/*
560.和为k的子数组(和两数之和的哈希解法非常类似，加了前缀和)
*/
func SubarraySum(nums []int, k int) int { //子串
	prefix_sum := 0                 //前缀和
	prefix_map := map[int]int{0: 1} //前缀哈希表记录前缀和出现次数
	res := 0
	for _, num := range nums {
		prefix_sum += num
		//sum(i,j)=prefix_sum[j]-prefix[i-1]
		//prefixSum[j] - prefixSum[i-1] == k
		//prefixSum[i-1] == prefixSum[j] - k
		//前面是否存在一个前缀和是 prefixSum - k，就能判断是否存在一个子数组之和为 k
		if v, ok := prefix_map[prefix_sum-k]; ok {
			res += v //有v个前缀和都等于，直接+v
		}
		prefix_map[prefix_sum] += 1 //记录当前的prefix_sum到哈希表中
	}
	return res
}

/*
53.最大子数组和
*/
func MaxSubArray(nums []int) int { //数组，动态规划方法
	//求出每一个以i下标结尾的数字的最大和fmax(i)
	//然后与max比较，把更大的赋值给max
	max := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i]+nums[i-1] > nums[i] {
			nums[i] = nums[i] + nums[i-1] //这里的nums[i]就存了当前下标结尾的数字的最大和
		}
		if max < nums[i] {
			max = nums[i]
		}
	}
	return max
}

/*
56.合并区间
*/
func Merge(intervals [][]int) [][]int { //数组
	if len(intervals) == 0 {
		return [][]int{}
	}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	// fmt.Println(intervals)
	res := make([][]int, 0)
	res = append(res, intervals[0])
	// fmt.Println(res)
	for i := 1; i < len(intervals); i++ {
		//说明一定可以合并
		if intervals[i][0] <= res[len(res)-1][1] {
			if intervals[i][1] > res[len(res)-1][1] {
				res[len(res)-1][1] = intervals[i][1]
			}
		} else { //无法合并
			res = append(res, intervals[i])
		}
	}
	return res
}

/*
189.轮转数组
*/
func Rotate1(nums []int, k int) { //数组 //第一种方法，使用额外空间，空间复杂度O(n)
	new_nums := make([]int, len(nums))
	for i := 0; i < len(nums); i++ {
		new_nums[(i+k)%len(nums)] = nums[i]
	}
	copy(nums, new_nums)
}

// 先封装原地翻转切片的方法
func reverse_list(nums []int) {
	for i := 0; i < len(nums)/2; i++ {
		nums[i], nums[len(nums)-i-1] = nums[len(nums)-i-1], nums[i]
	}
}

func Rotate2(nums []int, k int) { //数组 //第二种方法，不使用额外空间，空间复杂度O(1)
	k = k % len(nums)
	//借助翻转列表的思想
	reverse_list(nums)
	reverse_list(nums[0:k])
	reverse_list(nums[k:])
}

/*
238.除自身以外数组的乘积
*/
func ProductExceptSelf1(nums []int) []int { //数组 空间复杂度O(n)
	//方法一，构造L和R两个数组用来存储前序和后序的乘积
	L := make([]int, len(nums))
	R := make([]int, len(nums))
	L[0] = 1
	R[len(nums)-1] = 1
	res := make([]int, len(nums))
	for i := 1; i < len(nums); i++ {
		L[i] = L[i-1] * nums[i-1]
	}
	for j := len(nums) - 2; j > 0; j-- {
		R[j] = R[j+1] * nums[j+1]
	}
	for k := 0; k < len(nums); k++ {
		res[k] = L[k] * R[k]
	}
	return res
}

func ProductExceptSelf2(nums []int) []int { //数组 空间复杂度O(1)
	//不采用分别创建L和R来记录前序和后序乘积的方式
	//直接使用一个结果数组来记录，先记录前序，然后再计算后序
	length := len(nums)
	res := make([]int, length)
	res[0] = 1
	//先计算好前序
	for i := 1; i < length; i++ {
		res[i] = res[i-1] * nums[i-1]
	}
	//nums:[1,2,3,4]
	//res:[1,1,2,6]
	//res:[,8,6]
	R := 1
	for j := len(nums) - 1; j >= 0; j-- {
		res[j] = res[j] * R
		R = R * nums[j]
	}
	return res
}

/*
73.矩阵置零
*/
func SetZeroes1(matrix [][]int) { //矩阵 第一种解法 空间复杂度为O(m+n)
	//用row和col记录哪一行和哪一列需要置零
	row := make([]bool, len(matrix))
	col := make([]bool, len(matrix[0]))
	for i, r := range matrix {
		for j, v := range r {
			if v == 0 {
				row[i] = true
				col[j] = true
			}
		}
	}
	for i, r := range matrix {
		for j := range r {
			if row[i] || col[j] {
				r[j] = 0
			}
		}
	}
}

func SetZeroes2(matrix [][]int) { //矩阵 第二种解法 空间复杂度为O(1)
	//直接使用第一行和第一列作为标记，记录哪一行和哪一列需要置零
	//这种方法还需要考虑第一行和第一列是否本来就有0
	row_0, col_0 := false, false
	//先判断第一行和第一列是否本来就有0
	for i := 0; i < len(matrix); i++ { //判断列
		if matrix[i][0] == 0 {
			col_0 = true
		}
	}
	for j := 0; j < len(matrix[0]); j++ { //判断行
		if matrix[0][j] == 0 {
			row_0 = true
		}
	}
	//从1开始索引
	for i := 1; i < len(matrix); i++ {
		for j := 1; j < len(matrix[0]); j++ {
			if matrix[i][j] == 0 {
				matrix[i][0] = 0
				matrix[0][j] = 0
			}
		}
	}
	//置零
	for i := 1; i < len(matrix); i++ {
		for j := 1; j < len(matrix[0]); j++ {
			if matrix[i][0] == 0 || matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}
	}
	if row_0 {
		for i := 0; i < len(matrix[0]); i++ {
			matrix[0][i] = 0
		}
	}
	if col_0 {
		for j := 0; j < len(matrix); j++ {
			matrix[j][0] = 0
		}
	}
}

/*
54.螺旋矩阵
*/
func SpiralOrder(matrix [][]int) []int { //矩阵
	row := len(matrix)
	col := len(matrix[0])
	directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}} //对应右下左上
	//visited记录遍历过的位置
	visited := make([][]bool, row)
	for k := range visited {
		visited[k] = make([]bool, col)
	}
	res := make([]int, 0, row*col)
	i, j := 0, 0
	//初始方向为右
	dir := 0
	//列超出最大列，往下移动
	//行超出最大行，往左移动
	//列超出最小列，往上移动
	//行超出最小行，往右移动
	//如果在后期移动的过程中碰到了之前已经遍历过的元素，那么也要改变方向，规律为：
	//下->左
	//上->右
	//右->下
	//左->上
	for k := 0; k < row*col; k++ {
		visited[i][j] = true
		res = append(res, matrix[i][j])
		next_i, next_j := i+directions[dir][0], j+directions[dir][1]
		if next_j >= col || next_i >= row || next_j < 0 || next_i < 0 || visited[next_i][next_j] {
			dir = (dir + 1) % 4
			next_i, next_j = i+directions[dir][0], j+directions[dir][1]
		}
		i, j = next_i, next_j
	}
	return res
}

/*
48.旋转图像
*/
func Rotate11(matrix [][]int) { //矩阵，使用额外空间，空间复杂度O(n2)
	//先创建好一个结果矩阵
	res := make([][]int, len(matrix))
	for k := range res {
		res[k] = make([]int, len(matrix[0]))
	}
	//找规律
	//对于第一行，旋转过后出现在了倒数第一列的位置，并且第一行的第一个就是倒数第一列的第一个
	//对于第二行，旋转过后出现在了倒数第二列的位置，并且第二行的第一个就是倒数第二列的第一个
	//所以matrix[0][3]=matrix[0][0],matrix[1][3]=matrix[0][1],matrix[2][3]=matrix[0][2],matrix[3][3]=matrix[0][3]
	//matrix[j][len(matrix)-i-1]=matrix[i][j]
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			res[j][len(matrix)-i-1] = matrix[i][j]
		}
	}
	copy(matrix, res)
}

func Rotate22(matrix [][]int) { //矩阵，空间复杂度O(1)
	//先水平翻转，再按对角线翻转，即可得到顺时针旋转90度后的矩阵
	// 水平翻转 matrix[i][j] -> matrix[n-i-1][j]
	// 对角线翻转 matrix[i][j] -> matrix[j][i]
	// 联立两式，即可得 matrix[i][j] -> matrix[j][n-i-1],和上面找规律得到的式子是一样的
	n := len(matrix)
	for i := 0; i < n/2; i++ {
		for j := 0; j < n; j++ {
			matrix[i][j], matrix[n-i-1][j] = matrix[n-i-1][j], matrix[i][j]
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}

/*
240.搜索二维矩阵
*/
func SearchMatrix1(matrix [][]int, target int) bool { //矩阵 第一种解法，对每一行执行二分查找，时间复杂度O(mlogn)空间复杂度O(1)
	for _, row := range matrix {
		//返回第一个满足 row[i] >= target 的下标 i
		//如果不存在满足条件的元素（即所有元素都 < target），则返回 len(row)
		i := sort.SearchInts(row, target)
		if i < len(row) && row[i] == target { //判断条件不能调换顺序，否则会出现越界错误
			return true
		}
	}
	return false
}

func SearchMatrix2(matrix [][]int, target int) bool { //矩阵 第二种解法，Z字形查找
	//先从最右上角的元素开始
	//如果当前元素=target，直接返回
	//如果当前元素小于target，说明这一行都比target小，移动到下一行
	//如果当前元素大于target，说明这一列都比target大，移动到前一列
	row := len(matrix)
	col := len(matrix[0])
	i := 0
	j := col - 1
	for i < row && j >= 0 {
		if matrix[i][j] == target {
			return true
		} else if matrix[i][j] < target {
			i++
		} else {
			j--
		}
	}
	return false
}

/*
160.相交链表
*/

// Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

func GetIntersectionNode1(headA, headB *ListNode) *ListNode { //链表 方法一 空间复杂度O(max(m,n))
	visited := map[*ListNode]bool{}
	for headA != nil {
		visited[headA] = true
		headA = headA.Next
	}
	//检查headB指向的节点是否在visited中出现过
	for headB != nil {
		if visited[headB] {
			fmt.Printf("Intersected at '%d'", headB.Val)
			return headB
		}
		headB = headB.Next
	}
	fmt.Println("No intersection")
	return nil
}

func GetIntersectionNode2(headA, headB *ListNode) *ListNode { //链表 方法二 空间复杂度O(1)
	//链表A长度 a+c 链表B长度 b+c
	//都走过a+b+c后，要么相交，要么没有交点

	//我吹过你吹过的晚风，这算不算相拥？
	pa := headA
	pb := headB
	for pa != pb {
		if pa == nil {
			pa = headB
		} else {
			pa = pa.Next
		}
		if pb == nil {
			pb = headA
		} else {
			pb = pb.Next
		}
	}
	return pa
}

/*
206.反转链表
*/
func ReverseList(head *ListNode) *ListNode { //链表  迭代
	//定义一个prev用来记录当前节点的前一个节点
	var prev *ListNode
	curr := head
	for curr != nil {
		next := curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}
	return prev
}

/*
234.回文链表
*/
func IsPalindrome1(head *ListNode) bool { //链表
	//不可取的想法（因为逆置了之后，原链表已经消失了，所以无法跟新的链表比较）
	//先逆置链表，空间复杂度O(1)
	//再遍历比较每个元素值是否相同，若相同，则是回文链表

	//解法一
	//双指针+数组 空间复杂度O(n)
	nums := make([]int, 0)
	for head != nil {
		nums = append(nums, head.Val)
		head = head.Next
	}
	i, j := 0, len(nums)-1
	for i < j {
		if nums[i] != nums[j] {
			return false
		} else {
			i++
			j--
		}
	}
	return true
}

func IsPalindrome2(head *ListNode) bool { //链表
	if head == nil {
		return true
	}
	//解法二
	//空间复杂度O(1)
	//第一步，先找到前半部分的尾节点（快慢指针）
	fast_pointer := head
	slow_pointer := head
	for fast_pointer.Next != nil && fast_pointer.Next.Next != nil {
		fast_pointer = fast_pointer.Next.Next
		slow_pointer = slow_pointer.Next
	}
	//第二步，翻转后半部分链表
	var prev *ListNode
	curr := slow_pointer.Next
	for curr != nil {
		next := curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}
	//第三步，将前半部分与后半部分进行比较
	for prev != nil {
		if prev.Val != head.Val {
			return false
		} else {
			prev = prev.Next
			head = head.Next
		}
	}
	return true
}

/*
141.环形链表表
*/
func HasCycle1(head *ListNode) bool { //链表 方法一哈希 空间复杂度O(n)
	visited := map[*ListNode]int{}
	visited[head] += 1
	for head != nil {
		head = head.Next
		visited[head] += 1
		if visited[head] > 1 {
			return true
		}
	}
	return false
}

func HasCycle2(head *ListNode) bool { //链表 方法二 快慢指针 空间复杂度O(1)
	//龟兔赛跑，如果有环，终能相遇
	if head == nil {
		return false
	}
	slow_pointer := head
	fast_pointer := head.Next
	for slow_pointer != fast_pointer {
		if fast_pointer == nil || fast_pointer.Next == nil {
			return false
		}
		slow_pointer = slow_pointer.Next
		fast_pointer = fast_pointer.Next.Next
	}
	return true
}

/*
142.环形链表Ⅱ
*/
func DetectCycle1(head *ListNode) *ListNode { //链表 方法一 哈希 空间复杂度O(n)
	visited := map[*ListNode]int{}
	for head != nil {
		visited[head] += 1
		head = head.Next
		if visited[head] > 1 {
			return head
		}
	}
	return nil
}

func DetectCycle2(head *ListNode) *ListNode { //链表 方法二 快慢指针 空间复杂度O(1)
	//这种解法需要画图分析证明
	if head == nil {
		return nil
	}
	slow, fast := head, head
	for fast != nil {
		slow = slow.Next
		if fast.Next == nil {
			return nil
		}
		fast = fast.Next.Next
		if fast == slow {
			ptr := head
			for slow != ptr {
				slow = slow.Next
				ptr = ptr.Next
			}
			return ptr
		}
	}
	return nil
}

/*
21.合并两个有序链表
*/
func MergeTwoLists1(list1 *ListNode, list2 *ListNode) *ListNode { //方法一
	var res *ListNode
	l1, l2 := list1, list2
	for l1 != nil && l2 != nil {
		if l1.Val >= l2.Val {
			l2 = l2.Next
			list2.Next = res
			res = list2
			list2 = l2
		} else {
			l1 = l1.Next
			list1.Next = res
			res = list1
			list1 = l1
		}
	}
	for l1 != nil {
		l1 = l1.Next
		list1.Next = res
		res = list1
		list1 = l1
	}
	for l2 != nil {
		l2 = l2.Next
		list2.Next = res
		res = list2
		list2 = l2
	}
	var prev *ListNode
	curr := res
	for curr != nil {
		next := curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}
	return prev
}

func MergeTwoLists2(list1 *ListNode, list2 *ListNode) *ListNode { //方法二
	dummy := &ListNode{} //这里声明一个dummy，他并不是空指针，而是指向默认值val=0，next=nil的结构体的一个指针
	cur := dummy         //让cur指向和dummy相同的地址，cur往后移动并不会改变dummy
	for list1 != nil && list2 != nil {
		if list1.Val >= list2.Val {
			cur.Next = list2
			list2 = list2.Next
			cur = cur.Next
		} else {
			cur.Next = list1
			list1 = list1.Next
			cur = cur.Next
		}
	}
	for list1 != nil {
		cur.Next = list1
		list1 = list1.Next
		cur = cur.Next
	}
	for list2 != nil {
		cur.Next = list2
		list2 = list2.Next
		cur = cur.Next
	}
	return dummy.Next //头结点
}

/*
2.两数相加
*/
func AddTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode { //链表
	var val, jinwei int
	if l1.Val+l2.Val >= 10 {
		val = (l1.Val + l2.Val) % 10
		jinwei = 1
	} else {
		val = l1.Val + l2.Val
		jinwei = 0
	}
	l1 = l1.Next
	l2 = l2.Next
	res := &ListNode{Val: val, Next: nil}
	cur := res
	for l1 != nil && l2 != nil {
		if l1.Val+l2.Val >= 10 {
			if jinwei == 1 {
				val = (l1.Val + l2.Val + 1) % 10
				jinwei = 1
			} else {
				val = (l1.Val + l2.Val) % 10
				jinwei = 1
			}
		} else {
			if jinwei == 1 {
				val = (l1.Val + l2.Val + 1) % 10
				if val >= 10 || val == 0 {
					jinwei = 1
				} else {
					jinwei = 0
				}
			} else {
				val = l1.Val + l2.Val
				jinwei = 0
			}
		}
		new_node := &ListNode{Val: val, Next: nil}
		cur.Next = new_node
		cur = new_node
		l1 = l1.Next
		l2 = l2.Next
	}
	for l1 != nil {
		if l1.Val+jinwei == 10 {
			val = 0
			jinwei = 1
		} else {
			val = l1.Val + jinwei
			jinwei = 0
		}
		new_node := &ListNode{Val: val, Next: nil}
		cur.Next = new_node
		cur = new_node
		l1 = l1.Next
	}
	for l2 != nil {
		if l2.Val+jinwei == 10 {
			val = 0
			jinwei = 1
		} else {
			val = l2.Val + jinwei
			jinwei = 0
		}
		new_node := &ListNode{Val: val, Next: nil}
		cur.Next = new_node
		cur = new_node
		l2 = l2.Next
	}
	if jinwei == 1 {
		new_node := &ListNode{Val: 1, Next: nil}
		cur.Next = new_node
		cur = new_node
	}
	return res
}

/*
删除链表的倒数第n个节点
*/
func RemoveNthFromEnd1(head *ListNode, n int) *ListNode { //链表 方法一 时间复杂度O(n) 空间复杂度O(1)
	//需要两次遍历
	//最简单的想法，因为链表无法提前预估长度，所以只能先遍历确定总长度
	//然后第二次遍历执行删除操作
	length := 0
	cur := 1
	p := head
	q := head
	for p != nil {
		length++
		p = p.Next
	}
	loc := length - n + 1
	if loc == 1 {
		return head.Next
	}
	for q != nil {
		if cur == loc-1 {
			q.Next = q.Next.Next
			break
		} else {
			q = q.Next
			cur++
		}
	}
	return head
}

func RemoveNthFromEnd2(head *ListNode, n int) *ListNode { //链表 方法二 时间复杂度O(n) 空间复杂度O(n)
	//只需一次遍历
	//用栈来记录当前遍历的元素
	//一次遍历完成后，弹出栈的第n个元素就是我们要删除的元素
	if head.Next == nil {
		return nil
	}
	p := head
	stack := make([]*ListNode, 0)
	for p != nil {
		stack = append(stack, p)
		p = p.Next
	}
	if n == len(stack) {
		return head.Next
	}
	stack = stack[0 : len(stack)-n]
	stack[len(stack)-1].Next = stack[len(stack)-1].Next.Next
	return head
}

func RemoveNthFromEnd3(head *ListNode, n int) *ListNode { //链表 方法三 时间复杂度O(n) 空间复杂度O(1)
	//时间复杂度O(n),空间复杂度O(1)
	//一次遍历，快慢指针
	//开始时快指针与慢指针之间间隔n个节点
	slow := &ListNode{}
	slow.Next = head
	fast := head
	for n > 0 {
		fast = fast.Next
		n--
	}
	//一种特殊情况此时fast已经为nil了，说明删除的就是第一个节点
	if fast == nil {
		slow.Next = slow.Next.Next
		return slow.Next
	}
	for fast != nil {
		slow = slow.Next
		fast = fast.Next
	}
	//这时slow就指向待删除节点的前一个结点
	slow.Next = slow.Next.Next
	return head
}

/*
24.两两交换链表中的节点
*/
func SwapPairs(head *ListNode) *ListNode { //链表 迭代
	//三个指针模拟迭代过程
	if head == nil {
		return nil
	}
	dummy := &ListNode{Val: 0, Next: head}
	p := head
	q := head.Next
	r := dummy

	//循环体内做的
	for p != nil && q != nil {
		p.Next = q.Next
		q.Next = p
		r.Next = q
		r = p
		p = r.Next
		if p == nil {
			break
		} else {
			q = p.Next
		}
	}
	return dummy.Next
}

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

/*
138.随机链表的复制
*/
func CopyRandomList(head *Node) *Node { //链表 迭代 空间复杂度O(1)
	if head == nil {
		return nil
	}
	//第一步，复制每一个节点到该节点的后面
	p := head
	for p != nil {
		new_node := &Node{Val: p.Val}
		new_node.Next = p.Next
		p.Next = new_node
		p = p.Next.Next
	}
	//第二步，给复制节点的random指针赋值，依靠当前节点的next的random指针恰好等于当前节点的random指针的next
	p = head
	copy := p.Next
	for p != nil {
		if p.Random != nil {
			p.Next.Random = p.Random.Next
		} else {
			p.Next.Random = nil
		}
		if p.Next != nil {
			p = p.Next.Next
		}
	}
	//第三步，分离复制节点和源节点,并且保留源节点的结构
	old := head
	new := copy
	for old != nil {
		old.Next = old.Next.Next
		if new.Next != nil {
			new.Next = new.Next.Next
		}
		old = old.Next
		new = new.Next
	}
	return copy
}

/*
148.排序列表
*/
func SortList(head *ListNode) *ListNode { //链表 递归 归并排序
	if head == nil || head.Next == nil {
		return head
	}
	//1.快慢指针找mid
	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	mid := slow.Next
	slow.Next = nil //一定要有这一行，这样才能完全分开两个子链表
	//2.递归的排序
	left := SortList(head)
	right := SortList(mid)
	//3.合并两个有序链表
	return MergeTwoLists2(left, right)
}

/*
146.LRU缓存 记不住，并且还有一些没理解的地方方
*/
type DlinkedNode struct {
	Key, Value int
	Prev, Next *DlinkedNode
}

type LRUCache struct {
	Size       int
	Capacity   int
	Head, Tail *DlinkedNode
	Cache      map[int]*DlinkedNode
}

func InitDlinkedNode(key, value int) *DlinkedNode {
	return &DlinkedNode{
		Key:   key,
		Value: value,
	}
}

func Constructor(capacity int) LRUCache {
	l := LRUCache{
		Capacity: capacity,
		Head:     InitDlinkedNode(0, 0),
		Tail:     InitDlinkedNode(0, 0),
		Cache:    map[int]*DlinkedNode{},
	}
	l.Head.Next = l.Tail
	l.Tail.Prev = l.Head
	return l
}

// 新的缓存k-v来了，添加到LRUCache中去
// 双向链表的插入元素
func (this *LRUCache) AddToHead(node *DlinkedNode) {
	node.Prev = this.Head
	node.Next = this.Head.Next
	this.Head.Next.Prev = node
	this.Head.Next = node
}

// 被get函数调用k-v，返回value的同时，还需要把k-v移动到头部去
func (this *LRUCache) moveToHead(node *DlinkedNode) {
	this.removeNode(node)
	this.AddToHead(node)
}

// 双向链表的删除节点操作
func (this *LRUCache) removeNode(node *DlinkedNode) {
	node.Prev.Next = node.Next
	node.Next.Prev = node.Prev
}

// 超出LRU最大容量了，需要从尾部移除k-v
func (this *LRUCache) removeTail() *DlinkedNode {
	node := this.Tail.Prev
	this.removeNode(node)
	return node
}

func (this *LRUCache) Get(key int) int {
	if _, ok := this.Cache[key]; !ok {
		return -1
	}
	node := this.Cache[key]
	this.moveToHead(node)
	return node.Value
}

func (this *LRUCache) Put(key int, value int) {
	//LRU中没有这个节点
	if _, ok := this.Cache[key]; !ok {
		node := InitDlinkedNode(key, value)
		this.Cache[key] = node
		this.AddToHead(node)
		this.Size++
		if this.Size > this.Capacity {
			removed := this.removeTail()
			delete(this.Cache, removed.Key)
			this.Size--
		}
	} else {
		node := this.Cache[key]
		node.Value = value
		this.moveToHead(node)
	}
}

/*
94.二叉树的中序遍历
*/
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func InorderTraversal(root *TreeNode) []int { //二叉树
	//颜色标记法，可适用于前序中序后序遍历
	white, grey := 0, 1
	res := make([]int, 0)
	type colornode struct {
		color int
		node  *TreeNode
	}
	stack := []colornode{{color: white, node: root}}
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[0 : len(stack)-1]
		if n.node == nil {
			continue
		}
		if n.color == white {
			stack = append(stack, colornode{white, n.node.Right})
			stack = append(stack, colornode{grey, n.node})
			stack = append(stack, colornode{white, n.node.Left})
		} else {
			res = append(res, n.node.Val)
		}
	}
	return res
}

/*
104.二叉树的最大深度
*/
func MaxDepth(root *TreeNode) int { //二叉树
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right == nil {
		return 1
	}
	return Max(MaxDepth(root.Left), MaxDepth(root.Right)) + 1
}

func Max(a, b int) int {
	if a >= b {
		return a
	} else {
		return b
	}
}

/*
226.翻转二叉树
*/
func InvertTree(root *TreeNode) *TreeNode { //二叉树
	if root == nil {
		return nil
	}
	root.Left, root.Right = InvertTree(root.Right), InvertTree(root.Left)
	return root
}

/*
101.对称二叉树
*/
func IsSymmetric(root *TreeNode) bool { //二叉树
	if root == nil {
		return true
	}
	return Ismirror(root.Left, root.Right)
}

// 判断左子树和右子树是否轴对称
func Ismirror(root1 *TreeNode, root2 *TreeNode) bool {
	if root1 == nil && root2 == nil {
		return true
	}
	if root1 == nil || root2 == nil {
		return false
	}
	return root1.Val == root2.Val && Ismirror(root1.Left, root2.Right) && Ismirror(root1.Right, root2.Left)
}

/*
543.二叉树的直径
*/
func DiameterOfBinaryTree(root *TreeNode) int { //二叉树
	//求当前节点的深度就是比较左子树与右子树哪个深度更大
	//求当前节点的直径就是将左子树的深度加上右子树的深度
	max_length := 0
	_ = DepthOfNodePlusDiameter(root, &max_length)
	return max_length
}

func DepthOfNodePlusDiameter(node *TreeNode, max_length *int) int {
	if node == nil {
		return 0
	}
	left_depth := DepthOfNodePlusDiameter(node.Left, max_length)
	right_depth := DepthOfNodePlusDiameter(node.Right, max_length)
	if *max_length < left_depth+right_depth {
		*max_length = left_depth + right_depth
	}
	return Max(left_depth, right_depth) + 1 //我们是在求左右子树深度的过程中去讨论以当前节点为根节点的最大直径的值
}

/*
102.二叉树的层序遍历
*/
func LevelOrder(root *TreeNode) [][]int { //二叉树
	if root == nil {
		return nil
	}
	res := [][]int{{root.Val}}
	visited := [][]*TreeNode{{root}}
	for len(visited) > 0 {
		node := visited[0] //每次弹出队列的第一个元素
		visited = visited[1:]
		intermediate_res := []int{}
		intermediate_Treenode := []*TreeNode{}
		for i := 0; i < len(node); i++ {
			if node[i].Left != nil {
				intermediate_Treenode = append(intermediate_Treenode, node[i].Left)
				intermediate_res = append(intermediate_res, node[i].Left.Val)
			}
			if node[i].Right != nil {
				intermediate_Treenode = append(intermediate_Treenode, node[i].Right)
				intermediate_res = append(intermediate_res, node[i].Right.Val)
			}
		}
		if len(intermediate_Treenode) > 0 {
			visited = append(visited, intermediate_Treenode)
			res = append(res, intermediate_res)
		}
	}
	return res
}

/*
108.将有序数组转化为二叉搜索树
*/
func SortedArrayToBST(nums []int) *TreeNode { //二叉树
	return Helper1(nums, 0, len(nums)-1)
}

func Helper1(nums []int, left int, right int) *TreeNode {
	if left > right {
		return nil
	}
	mid := (left + right) / 2
	root := &TreeNode{Val: nums[mid]}
	root.Left = Helper1(nums, left, mid-1)
	root.Right = Helper1(nums, mid+1, right)
	return root
}

/*
92.验证二叉搜索树
*/
func IsValidBST(root *TreeNode) bool { //二叉树
	return Helper2(root, math.MinInt64, math.MaxInt64)
}

func Helper2(root *TreeNode, left, right int) bool {
	if root == nil {
		return true
	}
	if root.Val <= left || root.Val >= right {
		return false
	}
	return Helper2(root.Left, left, root.Val) && Helper2(root.Right, root.Val, right)
}

/*
230.二叉搜索树中第k小的元素
*/
func KthSmallest(root *TreeNode, k int) int { //二叉树
	//方法一，中序遍历，将结果存到一个切片里，遍历切片的第k-1个元素就是要找的第k个元素
	//因为是二叉搜索树，所以中序遍历结果一定是升序的
	res := []int{}
	res = Inorder(root, &res)
	return res[k-1]
}

func Inorder(root *TreeNode, res *[]int) []int {
	if root == nil {
		return nil
	}
	Inorder(root.Left, res)
	*res = append(*res, root.Val)
	Inorder(root.Right, res)
	return *res
}

/*
199.二叉树的右视图
*/
func RightSideView(root *TreeNode) []int { //二叉树
	//层序遍历，用一个切片存储每一层遍历过的元素
	//输出每一层的最后一个元素，即为右视图
	if root == nil {
		return nil
	}
	levelorder_res := [][]int{{root.Val}}
	visited := [][]*TreeNode{{root}}
	for len(visited) > 0 {
		node_set := visited[0]
		visited = visited[1:]
		intermediate_vis := []*TreeNode{}
		intermediate_res := []int{}
		for i := 0; i < len(node_set); i++ {
			if node_set[i].Left != nil {
				intermediate_vis = append(intermediate_vis, node_set[i].Left)
				intermediate_res = append(intermediate_res, node_set[i].Left.Val)
			}
			if node_set[i].Right != nil {
				intermediate_vis = append(intermediate_vis, node_set[i].Right)
				intermediate_res = append(intermediate_res, node_set[i].Right.Val)
			}
		}
		if len(intermediate_vis) > 0 { //一定要注意这里的条件，否则会无限制的增长，因为前面没做对当前节点为nil的判定
			visited = append(visited, intermediate_vis)
			levelorder_res = append(levelorder_res, intermediate_res)
		}
	}
	res := []int{}
	for j := 0; j < len(levelorder_res); j++ {
		res = append(res, levelorder_res[j][len(levelorder_res[j])-1])
	}
	return res
}

/*
114.二叉树展开为链表
*/
func Flatten1(root *TreeNode) { //二叉树
	//迭代先序颜色遍历法
	//创建额外树节点
	//但是函数并没有给返回值，所以这个算法无法实现，但是我第一个想法并且写出来
	if root == nil {
		return
	}
	if root.Left == nil && root.Right == nil {
		return
	}
	white, grey := 0, 1
	intermediate_res := []int{}
	type Colornode struct {
		Node  *TreeNode
		Color int
	}
	r := Colornode{Node: root, Color: white}
	stack := []Colornode{r}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[0 : len(stack)-1]
		if node.Node == nil {
			continue
		}
		if node.Color == white {
			stack = append(stack, Colornode{Node: node.Node.Right, Color: white})
			stack = append(stack, Colornode{Node: node.Node.Left, Color: white})
			stack = append(stack, Colornode{Node: node.Node, Color: grey})
		} else {
			intermediate_res = append(intermediate_res, node.Node.Val)
		}
	}
	rt := &TreeNode{Val: intermediate_res[0], Left: nil, Right: nil}
	p := rt
	for i := 0; i < len(intermediate_res); i++ {
		new_treenode := &TreeNode{Val: intermediate_res[i], Left: nil, Right: nil}
		p.Right = new_treenode
		p = new_treenode
	}
	return
}

func Flatten2(root *TreeNode) { //二叉树
	//真正的原地算法 空间复杂度O(1)
	//从根节点开始，先把左子树接到右子树上
	//root=root.Right,循环上述操作
	for root != nil {
		//先找到左子树的最右边节点，用最右边节点去链接右子树
		//先记录当前的左子树，因为一会就找不到了
		if root.Left != nil {
			cur := root.Left
			p := cur
			for cur.Right != nil {
				cur = cur.Right
			}
			cur.Right = root.Right
			root.Right = p
			root.Left = nil
		}
		root = root.Right
	}
}

/*
105.从前序与中序遍历序列构造二叉树
*/
func BuildTree(preorder []int, inorder []int) *TreeNode { //二叉树
	//递归构建左子树与右子树
	if len(preorder) == 0 {
		return nil
	}
	root := &TreeNode{Val: preorder[0], Left: nil, Right: nil}
	i := 0 //必须提前声明i，不然无法在下面的递归中传递长度这个参数
	for ; i < len(inorder); i++ {
		if inorder[i] == preorder[0] {
			break
		}
	}
	root.Left = BuildTree(preorder[1:i+1], inorder[0:i])
	root.Right = BuildTree(preorder[i+1:], inorder[i+1:])
	return root
}

/*
437.路经总和
*/
func PathSum(root *TreeNode, targetSum int) int { //二叉树
	//哈希表记录前缀和
	//递归回溯
	hashmap := map[int]int{0: 1}
	return DFS(root, 0, targetSum, hashmap)
}

func DFS(root *TreeNode, cur_sum int, target int, hashmap map[int]int) int {
	if root == nil {
		return 0
	}
	val := root.Val
	cur_sum += val
	count := hashmap[cur_sum-target]
	hashmap[cur_sum]++
	count += DFS(root.Left, cur_sum, target, hashmap)
	count += DFS(root.Right, cur_sum, target, hashmap)
	hashmap[cur_sum]--
	return count
}

/*
236.二叉树的最近公共祖先
*/
func LowestCommonAncestor(root, p, q *TreeNode) *TreeNode { //二叉树
	//递归
	if root == nil {
		return nil
	}
	if root.Val == p.Val || root.Val == q.Val {
		return root
	}
	Left := LowestCommonAncestor(root.Left, p, q)
	Right := LowestCommonAncestor(root.Right, p, q)
	if Left != nil && Right != nil {
		return root
	} else if Left == nil {
		return Right
	} else {
		return Left
	}
}

/*
200.岛屿数量
*/
func NumIslands(grid [][]byte) int { //图论
	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				DFS2(grid, i, j)
				count += 1
			}
		}
	}
	return count
}

func IsOutOfRange(grid [][]byte, x, y int) bool {
	return x < 0 || x >= len(grid) || y < 0 || y >= len(grid[0])
}

// 关于岛屿类问题的通用解法模板，只需要在主函数中根据实际需求添加部分代码
func DFS2(grid [][]byte, x, y int) {
	if IsOutOfRange(grid, x, y) {
		return
	}
	if grid[x][y] != '1' {
		return
	}
	grid[x][y] = '2'
	DFS2(grid, x-1, y)
	DFS2(grid, x+1, y)
	DFS2(grid, x, y-1)
	DFS2(grid, x, y+1)
}

/*
994.腐烂的橘子
*/
func OrangesRotting(grid [][]int) int {
	//遍历网格，计算新鲜橘子的个数，以及把所有腐烂的橘子入队
	//通过BFS循环，如果到最后新鲜橘子个数仍大于0，返回-1，否则返回记录的分钟数
	type Loc struct {
		X int
		Y int
	}
	queue := make([]Loc, 0)
	count := 0
	minute := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == 1 {
				count++
			} else if grid[i][j] == 2 {
				loc := Loc{i, j}
				queue = append(queue, loc)
			}
		}
	}
	//BFS循环条件
	//1.队列长度大于0
	//2.count大于0
	for count > 0 && len(queue) > 0 {
		length := len(queue)
		for i := 0; i < length; i++ {
			n := queue[0]
			queue = queue[1:]
			if n.X+1 < len(grid) && grid[n.X+1][n.Y] == 1 {
				count--
				grid[n.X+1][n.Y] = 2
				loc := Loc{n.X + 1, n.Y}
				queue = append(queue, loc)
			}
			if n.X-1 >= 0 && grid[n.X-1][n.Y] == 1 {
				count--
				grid[n.X-1][n.Y] = 2
				loc := Loc{n.X - 1, n.Y}
				queue = append(queue, loc)
			}
			if n.Y+1 < len(grid[0]) && grid[n.X][n.Y+1] == 1 {
				count--
				grid[n.X][n.Y+1] = 2
				loc := Loc{n.X, n.Y + 1}
				queue = append(queue, loc)
			}
			if n.Y-1 >= 0 && grid[n.X][n.Y-1] == 1 {
				count--
				grid[n.X][n.Y-1] = 2
				loc := Loc{n.X, n.Y - 1}
				queue = append(queue, loc)
			}
		}
		minute++
	}
	if count > 0 {
		return -1
	}
	return minute
}

/*
207.课程表
*/
func CanFinish(numCourses int, prerequisites [][]int) bool { //图论
	//BFS的思想
	//根据prerequisites初始化每个节点的入度，之后再根据这个初始化入度为0的节点队列
	//维护一个map，这个map记录了一门课的后置有哪些课
	in_degree := make([]int, numCourses)
	graph := map[int][]int{}

	for i := 0; i < len(prerequisites); i++ {
		x, y := prerequisites[i][0], prerequisites[i][1]
		in_degree[x]++
		graph[y] = append(graph[y], x)
	}

	queue := []int{}
	for i := 0; i < numCourses; i++ {
		if in_degree[i] == 0 {
			queue = append(queue, i)
		}
	}

	for len(queue) > 0 {
		course := queue[0]
		queue = queue[1:]
		for i := 0; i < len(graph[course]); i++ {
			in_degree[graph[course][i]]--
			if in_degree[graph[course][i]] == 0 {
				queue = append(queue, graph[course][i])
			}
		}
	}

	for i := 0; i < len(in_degree); i++ {
		if in_degree[i] > 0 {
			return false
		}
	}
	return true
}

/*
208.实现前缀树
*/
type Trie struct {
	IsEnd    bool
	children [26]*Trie
}

func Constructor2() Trie {
	return Trie{}
}

func (this *Trie) Insert(word string) { //图论
	node := this
	for _, byte := range word {
		index := byte - 'a'
		if node.children[index] == nil {
			node.children[index] = &Trie{}
		}
		node = node.children[index]
	}
	node.IsEnd = true
}

func (this *Trie) SearchPrefix(prefix string) *Trie {
	node := this
	for _, byte := range prefix {
		byte = byte - 'a'
		if node.children[byte] == nil {
			return nil
		}
		node = node.children[byte]
	}
	return node
}

func (this *Trie) Search(word string) bool {
	node := this.SearchPrefix(word)
	if node != nil && node.IsEnd == true {
		return true
	}
	return false
	// return this.SearchPrefix(word)!=nil&&this.SearchPrefix(word).IsEnd==true
}

func (this *Trie) StartsWith(prefix string) bool {
	node := this.SearchPrefix(prefix)
	if node != nil {
		return true
	}
	return false
}

/*
46.全排列
*/
func Permute(nums []int) [][]int { //回溯
	//定义结果数组，中间结果数组，used数组
	res := [][]int{}
	intermediate_res := []int{}
	used := map[int]bool{}

	var traceback func()
	traceback = func() {
		//如果当前中间结果的长度已经和nums长度相等，说明已经是一个排列了，添加到结果中去
		if len(intermediate_res) == len(nums) {
			//这里要注意要把intermediate_res复制一份再添加
			//否则在后面回溯的时候res的值也会发生改变
			//因为切片是引用类型
			res = append(res, append([]int{}, intermediate_res...))
		}

		for i := 0; i < len(nums); i++ {
			if used[i] {
				continue
			}
			used[i] = true
			intermediate_res = append(intermediate_res, nums[i])
			traceback()
			intermediate_res = intermediate_res[:len(intermediate_res)-1]
			used[i] = false
		}
	}
	traceback()
	return res
}

/*
35.搜索插入位置
*/
func SearchInsert(nums []int, target int) int { //二分查找
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (left + right) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return left
}

/*
74.搜索二维矩阵
*/
func SearchMatrix(matrix [][]int, target int) bool { //二分查找
	//两次二分查找
	//先确定在哪一行
	//再确定在这一行中元素是否存在
	//用到sort.Search和sort.SearchInts两个库方法

	/*
	   1.sort.Search(n int, f func(int) bool) int
	   返回 [0, n) 中第一个 f(i)==true 的下标
	   若全部为 false，则返回 n

	   2. sort.SearchInts(a []int, x int) int
	   等价于 sort.Search(len(a), func(i int) bool { return a[i] >= x })
	*/
	row := sort.Search(len(matrix), func(i int) bool {
		return matrix[i][0] > target
	}) - 1

	if row < 0 {
		return false
	}

	res := sort.SearchInts(matrix[row], target)
	if res < len(matrix[row]) && matrix[row][res] == target {
		return true
	}
	return false
}

/*
34. 在排序数组中查找元素的第一个和最后一个位置
*/
func SearchRange(nums []int, target int) []int { //二分查找
	//直接用sort.SearchInts()
	//开始位置即为第一个大于等于x的位置
	//结束位置即为第一个大于等于x+1的位置-1
	left := sort.SearchInts(nums, target)
	if left == len(nums) || nums[left] != target { //需要判断找到的下标是否符合要求
		return []int{-1, -1}
	}
	right := sort.SearchInts(nums, target+1) - 1 //关于right的情况就不要单独判断，因为right肯定符合要求
	return []int{left, right}
}

/*
33. 搜索旋转排序数组
*/
func Search(nums []int, target int) int { //二分查找
	//旋转过后也可用二分查找
	//旋转过后的数组一定是一部分有序，一部分无序
	//每次二分查找都是去有序的那部分去找
	//如果target不在有序的这部分，再去接着二分就有变成一部分有序，一部分无序
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (left + right) / 2
		if nums[mid] == target {
			return mid
		}
		if nums[left] <= nums[mid] { //如果左边有序
			if nums[left] <= target && target < nums[mid] { //target在左边部分
				right = mid - 1
			} else {
				left = mid + 1
			}
		} else { //右边有序
			if target > nums[mid] && target < nums[right] { //target在右边部分
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
	}
	return -1
}

/*
153. 寻找旋转排序数组中的最小值
*/
func FindMin(nums []int) int { //二分查找
	left, right := 0, len(nums)-1
	for left < right { //注意此题和上一题的循环条件有所不同，因为上一题他在数组中不一定存在，所以每个位置都要考虑到，而这一题一定存在
		mid := (left + right) / 2
		if nums[mid] > nums[right] { //最小值一定在右边
			left = mid + 1
		} else { //否则在左边
			right = mid
		}
	}
	return nums[left] //当循环结束，left==right说明已经找到了最小值
}

/*
20.有效的括号
*/
func IsValid(s string) bool { //栈
	stack := make([]rune, 0)
	for _, ch := range s {
		if ch == '(' || ch == '{' || ch == '[' {
			stack = append(stack, ch)
		} else {
			if len(stack) == 0 {
				return false
			}
			b := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if ch == '}' {
				if b != '{' {
					return false
				}
			} else if ch == ']' {
				if b != '[' {
					return false
				}
			} else {
				if b != '(' {
					return false
				}
			}
		}
	}
	return len(stack) == 0
}

/*
155.最小栈
*/
func min(x, y int) int { //栈
	if x < y {
		return x
	}
	return y
}

type MinStack struct {
	stack    []int
	minstack []int
}

func Constructor3() MinStack {
	return MinStack{
		stack:    []int{},
		minstack: []int{math.MaxInt64},
	}
}

func (this *MinStack) Push(val int) {
	this.stack = append(this.stack, val)
	top := this.minstack[len(this.minstack)-1]
	this.minstack = append(this.minstack, min(top, val))
}

func (this *MinStack) Pop() {
	this.stack = this.stack[:len(this.stack)-1]
	this.minstack = this.minstack[:len(this.minstack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
	return this.minstack[len(this.minstack)-1]
}

/*
394.字符串解码
*/
func DecodeString(s string) string { //栈
	type Helper struct {
		Multi int
		Ch    string
	}
	res := ""
	stack := []Helper{}
	multi := 0
	for _, ch := range s {
		if ch == '[' {
			stack = append(stack, Helper{
				Multi: multi,
				Ch:    res,
			})
			multi, res = 0, ""
		} else if ch == ']' {
			cur_multi, last_res := stack[len(stack)-1].Multi, stack[len(stack)-1].Ch
			stack = stack[:len(stack)-1]
			temp := ""
			for i := 0; i < cur_multi; i++ { //不能像python一样直接用数字乘字符表示几个字符，需要显式调用for循环来拼接
				temp += res
			}
			res = last_res + temp //顺序不能错
		} else if '0' <= ch && ch <= '9' {
			//从前往后从字符串顺序构造数字的方法，如123
			multi = multi*10 + int(ch-'0') //注意不能直接用int(ch)，int(ch)表示的是0-9的ascii码，ch-'0'才是对应的数字
		} else {
			res += string(ch)
		}
	}
	return res
}

/*
739.每日温度
*/
func DailyTemperatures(temperatures []int) []int { //栈
	//利用单调栈
	stack := []int{}
	res := make([]int, len(temperatures))
	for i := 0; i < len(temperatures); i++ {
		if len(stack) == 0 {
			stack = append(stack, i)
		} else {
			for len(stack) > 0 && temperatures[i] > temperatures[stack[len(stack)-1]] {
				res[stack[len(stack)-1]] = i - stack[len(stack)-1]
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, i)
		}
	}
	return res
}

/*
215. 数组中的第K个最大元素
*/
func FindKthLargest(nums []int, k int) int { //堆
	return QuickSelect(nums, 0, len(nums)-1, len(nums)-k)
}

// 实际用到的是改进版的快排，因为要实现O（n）时间复杂度
func QuickSelect(nums []int, l, r, k int) int {
	if l == r {
		return nums[k]
	}
	pivot := nums[l]
	i, j := l-1, r+1
	for i < j {
		for i++; nums[i] < pivot; i++ {
		}
		for j--; nums[j] > pivot; j-- {
		}
		if i < j {
			nums[i], nums[j] = nums[j], nums[i]
		}
	}
	if k <= j {
		return QuickSelect(nums, l, j, k)
	} else {
		return QuickSelect(nums, j+1, r, k)
	}
}

/*
347. 前 K 个高频元素
*/
func TopKFrequent(nums []int, k int) []int {
	//先创建一个哈希表存储num和freq的对应映射
	hashmap := make(map[int]int)
	for i := 0; i < len(nums); i++ {
		hashmap[nums[i]]++
	}
	//初始化堆，并把上面得到的hashmap对应的val和freq加入到堆中
	h := &MinHeap{}
	heap.Init(h)
	for num, freq := range hashmap {
		heap.Push(h, Element{num, freq})
		if h.Len() > k {
			heap.Pop(h)
		}
	}
	//返回结果
	res := []int{}
	n := h.Len() //细节，因为h.Len()在下面的循环中不断减小
	for i := 0; i < n; i++ {
		res = append(res, heap.Pop(h).(Element).Value)
	}
	return res
}

// 建堆，要实现Len，Swap，Push，Pop，Less五个接口
type Element struct {
	Value     int
	Frequency int
}

type MinHeap []Element

func (h MinHeap) Len() int {
	return len(h)
}

func (h MinHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h MinHeap) Less(i, j int) bool {
	return h[i].Frequency < h[j].Frequency //按频率排序
}

func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.(Element))
}

func (h *MinHeap) Pop() interface{} {
	res := (*h)[len(*h)-1]
	*h = (*h)[:len(*h)-1]
	return res
}

/*
121.买卖股票的最佳时机
*/
func MaxProfit(prices []int) int { //贪心算法
	max_profit := 0
	min_stock := math.MaxInt64
	//就算当前的price非常非常小，且后面的price比现在的price大不了多少，那么max_profit还是之前计算的那个最大的profit
	//就像是prices=[10,8,2,7,13,1,3],在更新了minstock后，max_profit依然是13-2，而不会是3-1
	for _, price := range prices {
		if price < min_stock {
			min_stock = price
		} else if price-min_stock > max_profit {
			max_profit = price - min_stock
		}
	}
	return max_profit
}

/*
55.跳跃游戏
*/
func CanJump(nums []int) bool { //贪心
	//对于nums，维护一个从当前元素最远可到达的位置即i+nums[i]
	//如果当前最远可到达的位置大于等于len(nums)-1,那么说明可以到最后一个位置
	//还需要保证当前的i是可以从前面跳到的
	//其他情况就不能到达
	max_dis := 0
	for i := 0; i < len(nums)-1; i++ {
		if i+nums[i] > max_dis && i <= max_dis {
			max_dis = i + nums[i]
		}
	}
	return max_dis >= len(nums)-1
}

/*
45.跳跃游戏II
*/
func Jump(nums []int) int { //贪心
	//从后往前找最小条约次数
	//每次都贪心的选择距离最后一个位置最远的位置，也就是最小的下标
	pos := len(nums) - 1
	count := 0
	for pos > 0 {
		for i := 0; i < pos; i++ {
			if nums[i]+i >= pos {
				pos = i
				count++
				break
			}
		}
	}
	return count
}

/*
763.划分字母区间
*/
func PartitionLabels(s string) []int { //贪心
	//先初始化一个哈希表，用于记录当前字符串每个字符最后出现的位置
	//准备切分，用一个start和end指针表示当前片段的开始和结尾
	//每次都拿当前字符的最后一次出现的位置与max_end作比较
	//当最后当前字符的位置已经等于max_end的时候，说明该片段所有的字符在后面都不会再出现，所以可以切分了
	hashmap := map[byte]int{}
	for index, ch := range s {
		hashmap[byte(ch)] = index //注意这里如果用for-range循环那么取到的ch是rune类型，如果用for循环，取到的就是byte类型
	}
	start, end := 0, 0
	max_end := 0
	res := []int{}
	for i := 0; i < len(s); i++ {
		if hashmap[s[i]] > max_end {
			max_end = hashmap[s[i]]
		}
		if i == max_end {
			res = append(res, end-start+1)
			start = end + 1
		}
		end++
	}
	return res
}

/*
70.爬楼梯
*/
func ClimbStairs(n int) int {
	//技巧的初始化方式
	//这种初始化方式可以保证f(1)正好等于1
	//p，q，r初始时分别代表f(-2),f(-1),f(0)
	p, q, r := 0, 0, 1
	for i := 1; i <= n; i++ {
		p = q
		q = r
		r = p + q
	}
	return r
}

/*
118.杨辉三角
*/
func Generate(numRows int) [][]int {
	// f(1,1)=1
	// f(2,1)=0+1=1=f(1,0)+f(1,1)
	// f(2,2)=1+0=1=f(1,1)+f(1,2)
	// f(3,1)=0+1=1=f(2,0)+f(2,1)
	// f(3,2)=1+1=2=f(2,1)+f(2,2)
	// f(3,3)=1+0=1=f(2,2)+f(2,3)
	// f(i,j)=f(i-1,j-1)+f(i-1,j)
	if numRows == 1 {
		return [][]int{{1}}
	}
	if numRows == 2 {
		return [][]int{{1}, {1, 1}}
	}
	res := [][]int{{1}, {1, 1}}
	for i := 3; i <= numRows; i++ {
		inter_res := []int{}
		inter_res = append(inter_res, 1)
		for j := 1; j <= i-2; j++ {
			inter_res = append(inter_res, res[i-2][j-1]+res[i-2][j])
		}
		inter_res = append(inter_res, 1)
		res = append(res, inter_res)
	}
	return res
}

/*
198.打家劫舍
*/
func Rob(nums []int) int {
	//如果nums的长度为0，那么输出就为0
	//如果长度为1，那么输出为max(nums[0],nums[1])
	//dp[i]=max(dp[i-2]+nums[i],dp[i-1])
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}
	if len(nums) == 2 {
		return max(nums[0], nums[1])
	}
	x, y := nums[0], max(nums[0], nums[1])
	res := 0
	for i := 2; i < len(nums); i++ {
		res = max(x+nums[i], y)
		x = y
		y = res
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

/*
279.完全平方数
*/
func NumSquares(n int) int { //数学 时间复杂度O（根号n） 空间复杂度O(1)
	//四平方和定理
	//任意一个正整数都可以被表示为至多四个正整数的平方和
	//当且仅当 n不等于4^k×(8m+7) 时，n 可以被表示为至多三个正整数的平方和。因此，当 n=4^k×(8m+7) 时，n 只能被表示为四个正整数的平方和
	if IsPerfectSquare(n) {
		return 1
	}
	if CheckIs4(n) {
		return 4
	}
	//返回2的情况就是这个数可以表示为a2+b2
	for i := 1; i*i <= n-1; i++ {
		a := n - i*i
		if IsPerfectSquare(a) {
			return 2
		}
	}
	return 3
}

// 判断是否是完全平方数
func IsPerfectSquare(y int) bool {
	x := int(math.Sqrt(float64(y)))
	if x*x == y {
		return true
	}
	return false
}

// 判断定理内容，即可以表示为四个正整数平方和的情况
func CheckIs4(n int) bool {
	for n%4 == 0 {
		n = n / 4
	}
	if (n-7)%8 == 0 {
		return true
	}
	return false
}

/*
322.零钱兑换
*/
func CoinChange(coins []int, amount int) int { //动态规划
	//不会了就去看csdn收藏了该题的解法
	F := make([]int, amount+1)
	for i := 0; i < len(F); i++ {
		F[i] = 9999999
	}
	F[0] = 0
	for i := 1; i < len(F); i++ {
		for _, coin := range coins {
			if i >= coin {
				F[i] = Min(F[i], F[i-coin]+1)
			}
		}
	}
	if F[amount] >= 9999999 {
		return -1
	}
	return F[amount]
}

func Min(a, b int) int {
	if a > b {
		return b
	}
	return a
}

/*
139.单词拆分
*/
func WordBreak(s string, wordDict []string) bool {
	//F[i]表示以当前字母结尾的字符串可否被wordDict中的字符串表示
	F := make([]bool, len(s)+1)
	for i := 0; i < len(F); i++ {
		F[i] = false
	}
	F[0] = true

	word_dict := make(map[string]bool, len(wordDict))
	for _, word := range wordDict {
		word_dict[word] = true
	}

	for i := 0; i < len(F); i++ { //开始指针
		for j := i + 1; j < len(F); j++ { //结束指针
			if F[i] && word_dict[s[i:j]] {
				F[j] = true
			}
		}
	}
	return F[len(F)-1]
}

/*
300.最长递增子序列
*/
func LengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	for i := 0; i < len(dp); i++ {
		dp[i] = 1
	}
	for i := 1; i < len(dp); i++ {
		max := 0
		for j := 0; j < i; j++ {
			if nums[i] <= nums[j] {
				continue
			} else {
				if dp[j] > max {
					max = dp[j]
				}
			}
			dp[i] = max + 1
		}
	}
	res := 0
	for _, v := range dp {
		if v > res {
			res = v
		}
	}
	return res
}

/*
152.乘积最大子数组
*/
func MaxProduct(nums []int) int {
	//每一个以nums[i]为结尾的乘积最大的非空连续子数组的值记为dp[i]
	//dp[i]=max(dp[i-1]*nums[i],nums[i])，因为nums[i]可能是负数
	//但是如果遇到特例，比如nums=[1,3,-2,9,-5]，那么上式会失效，因为存在偶数个负数
	//所以还需记录一个以nums[i]为结尾的乘积最小的非空连续子数组的值记为dp[i]
	if len(nums) == 0 {
		return 0
	}
	max_dp := nums[0]
	min_dp := nums[0]
	res := nums[0]
	for i := 1; i < len(nums); i++ {
		prev_max := max_dp
		prev_min := min_dp
		max_dp = max(nums[i], max(prev_max*nums[i], prev_min*nums[i]))
		min_dp = min(nums[i], min(prev_max*nums[i], prev_min*nums[i]))
		res = max(res, max_dp)
	}
	return res
}

/*
416.分割等和子集
*/
func CanPartition(nums []int) bool {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	if sum%2 != 0 {
		return false
	}
	target := sum / 2
	dp := make([]bool, target+1) //dp[i]表示当前的数字是否可以由几个数凑出来
	dp[0] = true
	for _, num := range nums {
		for i := target; i >= num; i-- {
			dp[i] = dp[i] || dp[i-num]
		}
	}
	return dp[target]
}

/*
136.只出现一次的数字
*/
func SingleNumber(nums []int) int {
	//任何数和 0 做异或运算，结果仍然是原来的数，即 a⊕0=a。
	//任何数和其自身做异或运算，结果是 0，即 a⊕a=0。
	//异或运算满足交换律和结合律，即 a⊕b⊕a=b⊕a⊕a=b⊕(a⊕a)=b⊕0=b
	//所以nums中的所有数进行异或操作得到的就是只出现一次的数字
	res := 0
	for _, num := range nums {
		res = res ^ num
	}
	return res
}

/*
169.多数元素
*/
func MajorityElement(nums []int) int {
	//摩尔投票法 因为多数元素的count一定大于其他元素的count
	//随机选择一个候选数candidate，维护candidate和count
	//如果当前数字==candidate，那么count++
	//否则count--，当count==0时，选择当前数字作为candidate
	count := 0
	var candidate int
	for _, num := range nums {
		if count == 0 {
			candidate = num
		}
		if num == candidate {
			count++
		} else {
			count--
		}
	}
	return candidate
}

/*
75.颜色分类
*/
func SortColors(nums []int) {
	p0, p1 := 0, 0 //指向0和指向1的指针
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			nums[p0], nums[i] = nums[i], nums[p0]
			if p0 < p1 { //说明已经有1被放到了前面
				nums[i], nums[p1] = nums[p1], nums[i]
			}
			p0++
			p1++ //p1也需要后移，因为如果不移动的话后面的1会和现在的0交换位置
		} else if nums[i] == 1 {
			nums[p1], nums[i] = nums[i], nums[p1]
			p1++
		}
	}
}

/*
31.下一个排列
*/
func NextPermutation(nums []int) {
	if len(nums) <= 1 {
		return
	}

	i, j, k := len(nums)-2, len(nums)-1, len(nums)-1

	// find: A[i]<A[j]
	for i >= 0 && nums[i] >= nums[j] {
		i--
		j--
	}

	if i >= 0 { // 不是最后一个排列
		// find: A[i]<A[k]
		for nums[i] >= nums[k] {
			k--
		}
		// swap A[i], A[k]
		nums[i], nums[k] = nums[k], nums[i]
	}

	// reverse A[j:end]
	for i, j := j, len(nums)-1; i < j; i, j = i+1, j-1 {
		nums[i], nums[j] = nums[j], nums[i]
	}
}

/*
287.寻找重复数
*/
func FindDuplicate(nums []int) int {
	slow := nums[0]
	fast := nums[0]
	for {
		slow = nums[slow]       //慢指针每次走一步
		fast = nums[nums[fast]] //快指针每次走两步
		if slow == fast {       //因为有环，快慢指针一定能相遇
			break
		}
	}
	//slow从头出发，fast从当前位置出发，再相遇的位置就是入环处
	slow = nums[0]
	for slow != fast {
		slow = nums[slow]
		fast = nums[fast]
	}
	return slow
}

/*
5.最长回文子串
*/
func LongestPalindrome(s string) string { //多维动态规划
	//二维动态规划
	//状态转移方程为s[i,j]=s[i]==s[j]&&s[i+1,j-1]
	//循环条件要注意i是从大到小，而j是从小到大
	n := len(s)
	if n < 2 {
		return s
	}
	dp := make([][]bool, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, n)
		dp[i][i] = true //初始化时所有当前单个字符都是回文
	}
	longest := 1
	start := 0
	for i := n - 1; i >= 0; i-- {
		for j := i + 1; j < n; j++ {
			if s[i] == s[j] && dp[i+1][j-1] || s[i] == s[j] && j-i == 1 {
				dp[i][j] = true
				if j-i+1 > longest {
					longest = j - i + 1
					start = i
				}
			}
		}
	}
	return s[start : start+longest]
}

/*
912.排序数组
*/
func SortArray(nums []int) []int { //手撕快排
	Quicksort(nums, 0, len(nums)-1)
	return nums
}

func Partition(nums []int, left, right int) int {
	pivot := nums[(left+right)/2]
	l, r := left-1, right+1
	for l < r {
		for l++; nums[l] < pivot; l++ {
		}
		for r--; nums[r] > pivot; r-- {
		}
		if l < r {
			nums[l], nums[r] = nums[r], nums[l]
		}
	}
	return r
}

func Quicksort(nums []int, left, right int) {
	if left >= right {
		return
	}
	pivot_index := Partition(nums, left, right)
	Quicksort(nums, left, pivot_index)
	Quicksort(nums, pivot_index+1, right)
}

/*
88.合并两个有序数组
*/
func Merge2(nums1 []int, m int, nums2 []int, n int) {
	i, j, k := m-1, n-1, m+n-1
	for i >= 0 && j >= 0 {
		if nums1[i] > nums2[j] {
			nums1[k] = nums1[i]
			i--
		} else {
			nums1[k] = nums2[j]
			j--
		}
		k--
	}
	for j >= 0 {
		nums1[k] = nums2[j]
		k--
		j--
	}
}

/*
103.二叉树的锯齿形层序遍历
*/
func ZigzagLevelOrder(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	if root.Left == nil && root.Right == nil {
		return [][]int{{root.Val}}
	}
	res := [][]int{{root.Val}}
	inter_res := []int{}
	queue := [][]*TreeNode{{root}}
	inter_queue := []*TreeNode{}
	for len(queue) > 0 {
		for i := 0; i < len(queue[0]); i++ {
			node := queue[0][i]
			if node.Left != nil {
				inter_queue = append(inter_queue, node.Left)
				inter_res = append(inter_res, node.Left.Val)
			}
			if node.Right != nil {
				inter_queue = append(inter_queue, node.Right)
				inter_res = append(inter_res, node.Right.Val)
			}
		}
		if len(inter_queue) > 0 {
			queue = append(queue, inter_queue)
			res = append(res, inter_res)
		}
		inter_queue = []*TreeNode{}
		inter_res = []int{}
		queue = queue[1:]
	}
	index := 0
	final_res := [][]int{}
	for i := 0; i < len(res); i++ {
		if index%2 == 0 {
			final_res = append(final_res, res[i])
		} else {
			reverseSlice(res[i])
			final_res = append(final_res, res[i])
		}
		index++
	}
	return final_res
}

// 原地逆序一个切片的方法
func reverseSlice(s []int) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}

/*
92.反转链表II
*/
func ReverseBetween(head *ListNode, left int, right int) *ListNode {
	curr := head
	dummy_node := &ListNode{}
	prev := dummy_node
	prev.Next = head
	index := 1
	for index < left {
		prev = curr
		curr = curr.Next
		index++
	}
	next := curr.Next
	for i := 0; i < right-left; i++ {
		curr.Next = next.Next
		next.Next = prev.Next
		prev.Next = next
		next = curr.Next
	}
	return dummy_node.Next
}

/*
415.字符串相加
*/
func AddStrings(num1 string, num2 string) string {
	jinwei := 0
	last_num1_idx := len(num1) - 1
	last_num2_idx := len(num2) - 1
	new_num := ""
	for last_num1_idx >= 0 || last_num2_idx >= 0 || jinwei > 0 {
		//这里巧妙地声明n1，n2，和下面的判断逻辑相呼应，就不要显式的补0了
		var n1, n2 int
		if last_num1_idx >= 0 {
			n1 = int(num1[last_num1_idx] - '0')
			last_num1_idx--
		}
		if last_num2_idx >= 0 {
			n2 = int(num2[last_num2_idx] - '0')
			last_num2_idx--
		}
		sum := n1 + n2 + jinwei
		jinwei = sum / 10
		sum = sum % 10
		new_num += string('0' + byte(sum))
	}
	//因为字符串是不可变的，所以要先转化成字符切片，逆置后再转换为string
	inter_res := []byte(new_num)
	for i, j := 0, len(new_num)-1; i < j; i, j = i+1, j-1 {
		inter_res[i], inter_res[j] = inter_res[j], inter_res[i]
	}
	new_num = string(inter_res)
	return new_num
}

/*
143.重排链表
*/
func middlenode(head *ListNode) *ListNode {
	slow := head
	fast := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	return slow
}

func reverselist(head *ListNode) *ListNode {
	p := &ListNode{}
	p.Next = nil
	q := head
	for q != nil {
		next := q.Next
		q.Next = p.Next
		p.Next = q
		q = next
	}
	return p.Next
}

func mergelist(l1, l2 *ListNode) *ListNode {
	dummy_node := &ListNode{}
	dummy_node.Next = l1
	p := l1
	q := l2
	next_p := p.Next
	next_q := q.Next
	for p != nil && q != nil {
		q.Next = p.Next
		p.Next = q
		p = next_p
		q = next_q
		if next_p != nil {
			next_p = next_p.Next
		}
		if next_q != nil {
			next_q = next_q.Next
		}
	}
	return dummy_node.Next
}

func ReorderList(head *ListNode) {
	if head.Next == nil || head.Next.Next == nil {
		return
	}
	//结果链表即为原链表左半边拼接逆序的原链表右半边
	//1.先找中间节点
	//2.逆序从中间节点开始的后半部分，注意逆序前要断开链表
	//3.合并链表
	Middle := middlenode(head)
	second := Middle.Next
	Middle.Next = nil
	q := reverselist(second)
	mergelist(head, q)
}

/*
1143.最长公共子序列
*/
func LongestCommonSubsequence(text1 string, text2 string) int { //二维动态规划
	//dp[i][j]表示text1的前i个字符与text2的前j个字符的最长公共子序列
	m := len(text1)
	n := len(text2)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if text1[i-1] == text2[j-1] { //一定注意这里是i-1和j-1，因为dp分别表示是前i个前j个，而字符串索引从0开始
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[m][n]
}

/*
82. 删除排序链表中的重复元素 II
*/
func DeleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	prev := &ListNode{}
	prev.Next = head
	dummy_node := prev
	p := head
	q := head.Next
	for q != nil {
		if p.Val == q.Val {
			for q != nil && q.Val == p.Val {
				q = q.Next
			}
			prev.Next = q
			p = q
			if q != nil {
				q = q.Next
			}
		} else {
			prev = p
			p = q
			q = q.Next
		}
	}
	return dummy_node.Next
}

/*
232. 用栈实现队列
*/
type MyQueue struct {
	stack1 []int
	stack2 []int
}

func Constructor4() MyQueue {
	my_queue := MyQueue{
		stack1: []int{},
		stack2: []int{},
	}
	return my_queue
}

func (this *MyQueue) Push(x int) {
	this.stack1 = append(this.stack1, x)
}

func (this *MyQueue) in2out() {
	for len(this.stack1) > 0 {
		this.stack2 = append(this.stack2, this.stack1[len(this.stack1)-1])
		this.stack1 = this.stack1[0 : len(this.stack1)-1]
	}
}

func (this *MyQueue) Pop() int {
	if len(this.stack2) == 0 {
		this.in2out()
	}
	x := this.stack2[len(this.stack2)-1]
	this.stack2 = this.stack2[0 : len(this.stack2)-1]
	return x
}

func (this *MyQueue) Peek() int {
	if len(this.stack2) == 0 {
		this.in2out()
	}
	return this.stack2[len(this.stack2)-1]
}

func (this *MyQueue) Empty() bool {
	return len(this.stack1) == 0 && len(this.stack2) == 0
}

/*
165. 比较版本号
*/
func CompareVersion(version1 string, version2 string) int {
	m, n := len(version1), len(version2)
	i, j := 0, 0
	for i < m || j < n {
		res1, res2 := 0, 0
		for ; i < m && version1[i] != '.'; i++ {
			res1 = res1*10 + int(version1[i]-'0')
			fmt.Println(res1)
		}
		i++
		for ; j < n && version2[j] != '.'; j++ {
			res2 = res2*10 + int(version2[j]-'0')
			fmt.Println(res2)
		}
		j++
		if res1 > res2 {
			return 1
		}
		if res2 > res1 {
			return -1
		}
	}
	return 0
}

/*
69. x 的平方根
*/
func MySqrt(x int) int {
	left, right := 1, x
	for left <= right {
		mid := (left + right) / 2
		if mid*mid == x {
			return mid
		} else if mid*mid < x {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return right
}

/*
8. 字符串转换整数 (atoi)
*/
func MyAtoi(s string) int {
	res := 0
	pos_flag := []int{}
	neg_flag := []int{}
	count := 0
	started := false
	for i := 0; i < len(s); i++ {
		if !started {
			if s[i] == ' ' {
				continue
			} else if s[i] == '-' && count == 0 {
				neg_flag = append(neg_flag, 0)
				started = true
			} else if s[i] == '+' && count == 0 {
				pos_flag = append(pos_flag, 1)
				started = true
			} else if s[i] >= '0' && s[i] <= '9' {
				res = int(s[i] - '0')
				started = true
				count += 1
			} else {
				break
			}
		} else {
			//不能在最后判断是否超过了int32的最大值，而是要在每一次计算res的时候判断
			//因为如果最后再判断可能res已经越界了，从正数变成负数，或者从负数变成正数
			if s[i] >= '0' && s[i] <= '9' {
				digit := int(s[i] - '0')
				sign := 1
				if len(neg_flag) > 0 {
					sign = -1
				}
				if res > math.MaxInt32/10 || (res == math.MaxInt32/10 && digit > 7) {
					if sign == 1 {
						return math.MaxInt32
					} else {
						return math.MinInt32
					}
				}
				res = res*10 + digit
			} else {
				break
			}
		}
	}
	if len(pos_flag) >= 1 && len(neg_flag) >= 1 {
		return 0
	}
	if len(neg_flag) > 0 {
		return -res
	}
	return res
}

/*
43. 字符串相乘
*/
func Multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	m, n := len(num1)-1, len(num2)-1
	ans := "0"
	for j := n; j >= 0; j-- {
		jinwei := 0
		curr := ""
		x := int(num2[j] - '0')
		//补零操作
		for k := 0; k < n-j; k++ {
			curr += "0"
		}
		for i := m; i >= 0; i-- {
			y := int(num1[i] - '0')
			res := x*y + jinwei
			jinwei = res / 10
			res = res % 10
			curr = strconv.Itoa(res) + curr
		}
		//每次乘法完后可能还有多余的进位，比如jinwei=12，这里一定在下面加上%10,否则就不是一位一位的加，而会多加
		for ; jinwei != 0; jinwei = jinwei / 10 {
			curr = strconv.Itoa(jinwei%10) + curr
		}
		ans = add(ans, curr)
	}
	return ans
}

// 返回两个字符串相加的结果
// 9999999
//
//	9999
func add(num1 string, num2 string) string {
	m, n := len(num1)-1, len(num2)-1
	res := []byte{}
	jinwei := 0
	for m >= 0 && n >= 0 {
		add_res := int(num1[m]-'0') + int(num2[n]-'0') + jinwei
		if add_res >= 10 {
			add_res = add_res % 10
			jinwei = 1
		} else {
			jinwei = 0
		}
		res = append(res, '0'+byte(add_res))
		m--
		n--
	}
	for m >= 0 {
		add_res := int(num1[m]-'0') + jinwei
		if add_res >= 10 {
			add_res = add_res % 10
			jinwei = 1
		} else {
			jinwei = 0
		}
		res = append(res, '0'+byte(add_res))
		m--
	}
	for n >= 0 {
		add_res := int(num2[n]-'0') + jinwei
		if add_res >= 10 {
			add_res = add_res % 10
			jinwei = 1
		} else {
			jinwei = 0
		}
		res = append(res, '0'+byte(add_res))
		n--
	}
	if jinwei != 0 {
		res = append(res, '1')
	}
	for i, j := 0, len(res)-1; i < j; i, j = i+1, j-1 {
		res[i], res[j] = res[j], res[i]
	}
	return string(res)
}

/*
151. 反转字符串中的单词
*/
func ReverseWords(s string) string {
	s_slice := []string{}
	i := 0
	for i < len(s) {
		if s[i] == ' ' {
			i++
			continue
		}
		if s[i] >= 'A' && s[i] <= 'z' || s[i] >= '0' && s[i] <= '9' {
			start := i
			for i < len(s) && s[i] != ' ' {
				i++
			}
			end := i
			s_slice = append(s_slice, s[start:end])
		}
	}
	res := ""
	for j := len(s_slice) - 1; j >= 0; j-- {
		res += s_slice[j]
		if j != 0 {
			res += " "
		}
	}
	return res
}

/*
78. 子集
*/
func Subsets(nums []int) [][]int {
	res := [][]int{}
	inter_res := []int{}
	var backtrack func(start int)
	backtrack = func(start int) {
		inter_res_copy := make([]int, len(inter_res))
		copy(inter_res_copy, inter_res)
		res = append(res, inter_res_copy)

		for i := start; i < len(nums); i++ {
			inter_res = append(inter_res, nums[i])
			backtrack(i + 1)
			inter_res = inter_res[:len(inter_res)-1]
		}
	}
	backtrack(0)
	return res
}

/*
22.括号生成
*/
func GenerateParenthesis(n int) []string {
	//左括号小于n就可以添加左括号
	//右括号小于左括号就可以添加右括号
	//其他情况都是非法的
	res := []string{}
	inter_res := []byte{}
	var backtrack func(left, right int)
	backtrack = func(left, right int) {
		//这里就不需要copy一个inter_res出来，因为我们不是直接把inter_res加到res里面，而是把字符串添加进去
		if len(inter_res) == 2*n {
			res = append(res, string(inter_res))
		}
		if left < n {
			inter_res = append(inter_res, '(')
			backtrack(left+1, right)
			inter_res = inter_res[:len(inter_res)-1]
		}
		if right < left {
			inter_res = append(inter_res, ')')
			backtrack(left, right+1)
			inter_res = inter_res[:len(inter_res)-1]
		}
	}
	backtrack(0, 0)
	return res
}

/*
129. 求根节点到叶节点数字之和
*/
func SumNumbers(root *TreeNode) int {
	num, sum := 0, 0
	var dfs func(root *TreeNode)
	dfs = func(root *TreeNode) {
		if root == nil {
			return
		}
		num = num*10 + root.Val
		if root.Left == nil && root.Right == nil {
			sum += num
		}
		dfs(root.Left)
		dfs(root.Right)
		num = (num - root.Val) / 10
	}
	dfs(root)
	return sum
}

/*
144. 二叉树先序遍历
*/
func PreorderTraversal1(root *TreeNode) []int { //非递归 颜色标记法 用栈
	if root == nil {
		return []int{}
	}
	white, grey := 0, 1
	type colornode struct {
		Node  *TreeNode
		Color int
	}
	vis := []colornode{{Node: root, Color: white}}
	res := []int{}
	for len(vis) > 0 {
		n := vis[len(vis)-1]
		vis = vis[:len(vis)-1]
		if n.Node == nil {
			continue
		}
		if n.Color == white {
			vis = append(vis, colornode{n.Node.Right, white})
			vis = append(vis, colornode{n.Node.Left, white})
			vis = append(vis, colornode{n.Node, grey})
		} else {
			res = append(res, n.Node.Val)
		}
	}
	return res
}

func PreorderTraversal2(root *TreeNode) []int { //递归
	if root == nil {
		return []int{}
	}
	res := &[]int{}
	var preorder func(root *TreeNode, res *[]int)
	preorder = func(root *TreeNode, res *[]int) {
		if root == nil {
			return
		}
		*res = append(*res, root.Val)
		if root.Left != nil {
			preorder(root.Left, res)
		}
		if root.Right != nil {
			preorder(root.Right, res)
		}
	}
	preorder(root, res)
	return *res
}

/*
134. 加油站
*/
func CanCompleteCircuit(gas []int, cost []int) int {
	length := len(gas)
	start := 0
	//外层循环控制从哪个加油站开始出发
	//内层循环控制当前走过了多少个加油站
	for start < length {
		sum_gas := 0
		sum_cost := 0
		count := 0
		for count < length {
			idx := (start + count) % length //想成一个环形，如果从中间出发，我们还需要回到前面
			sum_gas += gas[idx]
			sum_cost += cost[idx]
			if sum_cost > sum_gas {
				break
			}
			count++
		}
		if count == length {
			return start
		} else {
			start = start + count + 1 //直接跳过中间部分，因为从中间部分的任何一个位置开始肯定也到不了，体现出全局贪心，而不是局部贪心
		}
	}
	return -1
}

/*
40. 组合总和 II
*/
func CombinationSum2(candidates []int, target int) [][]int {
	res := &[][]int{}
	inter_res := []int{}
	start := 0
	sort.Ints(candidates)
	Dfs(candidates, target, inter_res, res, start)
	return *res
}

func Dfs(candidates []int, target int, inter_res []int, res *[][]int, start int) {
	if target == 0 {
		//一定要注意这里如果直接append inter_res，之后对inter_res的改动会影响之前res的结果，所以必须深拷贝一份
		inter_res_copy := make([]int, len(inter_res))
		copy(inter_res_copy, inter_res)
		*res = append(*res, inter_res_copy)
		return
	}
	for i := start; i < len(candidates); i++ {
		//剪枝，当前target减去选中的值已经小于0，不考虑
		if target-candidates[i] < 0 {
			break
		}
		//剪枝，当前选中数字和上一个选中的数字相等，不考虑，因为我们已经排好序了
		if i > start && candidates[i] == candidates[i-1] {
			continue
		}
		inter_res = append(inter_res, candidates[i])
		Dfs(candidates, target-candidates[i], inter_res, res, i+1)
		inter_res = inter_res[:len(inter_res)-1]
	}
}
