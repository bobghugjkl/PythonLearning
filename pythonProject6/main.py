n = int(input())
nums = list(map(int, input().split()))


def longest_consecutive(nums):
    if not nums:
        return 0
    nums.sort()
    n = len(nums)
    max_count = 1
    count = 1
    i = 1
    while i < n:
        if nums[i] == nums[i - 1] + 1:
            count += 1
        else:
            max_count = max(max_count, count)
            count = 1
        i += 1
    max_count = max(max_count, count)
    return max_count


print(longest_consecutive(nums))
