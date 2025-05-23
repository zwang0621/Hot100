package main

import (
	"Hot100_go/utils"
	"fmt"
)

func main() {
	// fmt.Println(utils.TwoSum([]int{3, 2, 4}, 6))
	// nums := []int{0, 1}
	// utils.MoveZeroes(nums)
	// fmt.Println(nums)
	// fmt.Println(utils.MaxArea([]int{1, 2, 4, 3}))
	// nums := []int{-1, 0, 1, 2, -1, -4}
	// fmt.Println(utils.ThreeSum(nums))
	// s := "cbaebabacd"
	// p := "abc"
	// fmt.Println(utils.LengthOfLongestSubstring(s))
	// fmt.Println(utils.FindAnagrams(s, p))
	// nums := []int{1, 1, 1}
	// k := 2
	// fmt.Println(utils.SubarraySum(nums, k))
	// nums := []int{1, 2, 3, 4, 5, 6, 7}
	// k := 3
	// utils.Rotate2(nums, k)
	// fmt.Println(nums)
	//fmt.Println(utils.ProductExceptSelf1([]int{1, 2, 3, 4}))
	//fmt.Println(utils.ProductExceptSelf2([]int{1, 2, 3, 4}))
	// matrix := [][]int{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}}
	// utils.SetZeroes2(matrix)
	// fmt.Println(matrix)
	// matrix := [][]int{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}
	// fmt.Println(utils.SpiralOrder(matrix))
	// head := &utils.ListNode{
	// 	Val: 1,
	// 	Next: &utils.ListNode{
	// 		Val: 2,
	// 		Next: &utils.ListNode{
	// 			Val: 3,
	// 			Next: &utils.ListNode{
	// 				Val:  4,
	// 				Next: nil,
	// 			},
	// 		},
	// 	},
	// }

	// fmt.Println(utils.IsPalindrome2(head))
	// fmt.Println(utils.SwapPairs(head))
	nums := []int{1, 3, 5, 6}
	target := 7
	fmt.Println(utils.SearchInsert(nums, target))
}
