Day 5 – Sliding Window (Fresh Start)
🔹 What Sliding Window is REALLY about

You use sliding window when:

The problem talks about contiguous subarray / substring

You want to optimize from O(n²) → O(n)

Core idea:

Maintain a window [left … right] and move it smartly.

🧩 Sliding Window – 3 Core Templates

You only need to remember 3 templates.

1️⃣ Template A: Fixed Size Window
Problem

Maximum sum subarray of size k

Thinking (no code yet)

Expand window to size k

Record answer

Slide: remove left, add right

Key invariant

Window size is always k
Add → Check size → Update answer → Remove → Move left

2️⃣ Template B: Variable Window (Condition on COUNT / SUM)
Problem

Smallest subarray with sum ≥ S

Thinking

Expand right until condition is met

Shrink left while condition holds

Track minimum length

Key invariant

Window sum controls validity
==========================================


Main algo-->

for each element as right:
    include element at right into window

    while window is invalid:
        remove element at left
        move left forward

    window is valid here
    update answer




=====================================

3️⃣ Template C: Variable Window (Distinct / Frequency Based)
Problem

Longest substring with at most k distinct characters

Thinking

Expand right

Track frequency with hashmap

Shrink when distinct > k

Key invariant

len(freq_map) ≤ k

🚦 Universal Sliding Window Rules (Memorize)

right always moves forward

left only moves forward

Inside a while, left must move every time

Update answer after window is valid

If you follow these, you won’t get stuck.








