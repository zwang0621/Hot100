package utils

import (
	"fmt"
	"sort"
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

/*
24.两两交换链表中的节点
*/
func SwapPairs(head *ListNode) *ListNode {

}
