def get_first_neg_ksub(arr,k):
    left = 0
    right = 0
    neg_list = []
    window = []
    # 1, -1, 2 -3
    for right in range (len(arr)):
        #do something
        if(arr[right] <0):
            #print("arr[right] is ", arr[right])
            window.append(arr[right])
        #add to list
        if((right-left+1) == k):
            #recored result in neg_list
            print("window full case ", window)
            if(len(window) == 0):
                neg_list.append(0)
            else:
                neg_list.append(window[0])
            
            if(len(window) != 0) and  (arr[ left] == window[0]) :
                print("going to pop", window[0])
                window.pop(0)
            left += 1
            #increment the left
    return neg_list



#Given an array of integers and an integer k, find the maximum sum of any contiguous subarray of size k.
def max_sum_subarray_k(arr, k):
    left = 0 
    right = 0
    max_sum = 0
    final_sum = 0
    print("given arr is " , arr)
    for right in range(len(arr)):
        max_sum = max_sum + arr[right]
        #when window is full
        if((right-left+1) == k):
            final_sum = max(final_sum , max_sum)
            max_sum -= arr[left]
            left += 1

    return final_sum


#calling main
if __name__ == "__main__" :
    arr = [1,2,3,4,5]
    #print("final sum is ",max_sum_subarray_k(arr,3))
    lst = [1,-1,2,-3,4,-5,6,-7]
    print("Input lst is ", lst)
    k =3 
    print("subarry length is ",k)
    print("list is ", get_first_neg_ksub(lst,k))
