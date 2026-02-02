#Two Sum (sorted)

#Reverse String

#Remove Duplicates (sorted)

#Container With Most Water

#Trapping Rain Water 


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

#max_height container problem
def max_height(height):
    low = 0
    n = len(height)
    high = n-1
    max_area = 0

    while low<high:
        width = high - low
        length = min(height[low], height[high])
        max_area = max(max_area, length*width)
        if height[low] <height[high]:
                     low += 1
        else:
                     high -= 1
    return max_area    

#  Given an array height where each bar has width 1,
#compute how much water can be trapped after raining.
def trap(arr):
     left = 0
     right = len(arr) -1 
     max_left = 0
     max_right = 0
     water = 0
     while left <right:
         if arr[left] < arr[right]:
             if arr[left] >= max_left:
                 max_left =  arr[left]
             else:
                  water += max_left - arr[left]
             left += 1
         else:
              if arr[right] >= max_right:
                  max_right = arr[right]
              else:
                  water += max_right - arr[right]
              right -= 1
     return water



##########################################main block##########################
if __name__ == "__main__" :
    #arr = [1,2,3,4,5]
   # s = 10
   # print("return indices are" , two_sum_sorted(arr,s))
    #str1 = "howru"
   # print("reversed string is", reverse_string(str1))
    #arr = [1,1,2,2,3,3,4,5]
    #print("no of non duplicate elements in array is ", remove_duplicate_sorted(arr))

    #height = [1,8,6,2,5,4,8,3,7]
    #print("max_area is ", max_height(height))
    arr = [0,1,0,2,1,0,1,3,2,1,2,1]
    print("max water trapped is ", trap(arr))
