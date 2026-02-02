🧠 Day 6 Recap — Two Pointers (Cheat Sheet)
1️⃣ What is the Two Pointers technique?

Use two indices to traverse a structure in a coordinated way
to reduce time complexity (usually O(n²) → O(n)).

2️⃣ When to use Two Pointers?

✔ Array or string
✔ Often sorted
✔ Need to compare pairs, ranges, or ends
✔ Want in-place solution
✔ Want to avoid nested loops

3️⃣ Two Core Pointer Patterns
🔹 Pattern A: Opposite Ends (meet in the middle)

Used when:

Array is sorted

Comparing two values

left → ← right


Template:

left = 0
right = n - 1

while left < right:
    if condition:
        left += 1
    else:
        right -= 1

🔹 Pattern B: Same Direction (slow + fast)

Used when:

Removing / compressing elements

Maintaining invariants

i (slow) → j (fast)


Template:

i = 0
for j in range(n):
    if valid:
        i += 1
        arr[i] = arr[j]

4️⃣ Problems You Mastered
✅ 1. Two Sum (Sorted Array)

Goal: Find indices where arr[i] + arr[j] == target

Rule:

Sum too small → move left

Sum too large → move right

if arr[left] + arr[right] < target:
    left += 1
else:
    right -= 1

✅ 2. Reverse String / Array

Goal: Reverse in place

Rule:

Swap ends

Move both pointers inward

arr[l], arr[r] = arr[r], arr[l]
l += 1
r -= 1

✅ 3. Remove Duplicates from Sorted Array

Goal: Compress array in place

Invariant:

Elements 0..i are unique

if arr[j] != arr[i]:
    i += 1
    arr[i] = arr[j]


Return i + 1.

✅ 4. Container With Most Water

Formula:

area = min(height[l], height[r]) * (r - l)


Golden Rule:

Always move the pointer pointing to the shorter line

✅ 5. Trapping Rain Water (Hard)

Formula at index i:

water = min(max_left, max_right) - height[i]


Key Insight:

Water is limited by the smaller boundary

Rule:

Process the smaller side

Maintain left_max & right_max

5️⃣ Key Invariants to Remember
Problem	Invariant
Two Sum	Sorted order gives direction
Reverse	Swap until pointers meet
Remove Duplicates	0..i always unique
Max Water	Shorter side limits area
Rain Water	Smaller boundary limits water
6️⃣ Common Mistakes to Avoid

❌ Using nested loops
❌ Moving wrong pointer
❌ Forgetting boundary updates
❌ Breaking invariant order
❌ Overthinking pointer movement

7️⃣ One Golden Day-6 Sentence

Two pointers work by eliminating impossible cases through directional movement.

