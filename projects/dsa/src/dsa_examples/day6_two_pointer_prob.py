def two_sum_sorted(arr, s):
    left = 0
    right = len(arr)-1
    total = s
    while left<right:
        if(arr[left]+arr[right]) == total:
            return (left,right)
        elif (arr[left] +arr[right]) < total:
            left += 1
        else:
            right -= 1
    return None

def reverse_string(s):
    s1 = list(s)
    left = 0
    right = len(s1) -1
    while(left < right):
        s1[left],s1[right] = s1[right], s1[left]
        left += 1
        right-= 1
    return "".join(s1)

def remove_duplicate_sorted(arr):
    i = 0
    print("len arr is" , len(arr))
    for j in range(1, len(arr)):
        if arr[i] != arr[j]:
            i += 1
            arr[i] = arr[j]
    print(" i is arr is " , i,arr)
    return i+1
if __name__ == "__main__" :
    #arr = [1,2,3,4,5]
   # s = 10
   # print("return indices are" , two_sum_sorted(arr,s))
    #str1 = "howru"
   # print("reversed string is", reverse_string(str1))
    arr = [1,1,2,2,3,3,4,5]
    print("no of non duplicate elements in array is ", remove_duplicate_sorted(arr))
