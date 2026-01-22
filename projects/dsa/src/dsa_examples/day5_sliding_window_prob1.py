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


### Longest substring of K distinct characters 
def longest_substring_k_distinct(s, k):
    print("inside fun s , k", s , k)
    left = 0
    right = 0
    window = {}
    result  = []
    max_len = 0
    for right in range (len(s)):
        #populate the window
        if s[right] not in window:
            window[s[right]] = 1
        else:
            window[s[right]] += 1

        #check window validity

        while len(window) > k:
            print("window invalid !!!")
            window[s[left]] -= 1
            if(window[s[left]] == 0):
                #remove from window
                del window[s[left]]
            left +=1 

        #update result
        result = window
        max_len = max(max_len,(right-left+1))
        print("max valid string is , len",result, max_len)


    return max_len

#longest substring with no repeating characters
def longest_substring_distinct(s):
    print("inside fun s ", s )
    left = 0
    right = 0
    window = {}
    result  = []
    max_len = 0
    for right in range (len(s)):
        #populate the window
        if s[right] not in window:
            window[s[right]] = 1
        else:
            window[s[right]] += 1

        #check window validity

        while window[s[right]] >1 :
            print("window invalid !!!")
            window[s[left]] -= 1
            if(window[s[left]] == 0):
                #remove from window
                del window[s[left]]
            left +=1 

        #update result
        result = window
        max_len = max(max_len,(right-left+1))
        print("max valid string is , len",result, max_len)


    return max_len



#calling main
if __name__ == "__main__" :
    arr = [1,2,3,4,5]
    #print("final sum is ",max_sum_subarray_k(arr,3))
   # lst = [1,-1,2,-3,4,-5,6,-7]
   # print("Input lst is ", lst)
    k =3
    #print("subarry length is ",k)
    #print("list is ", get_first_neg_ksub(lst,k))
    #s = "abcdbacbcadd"
    #print("max substring is ",longest_substring_k_distinct(s,k)) 
    s = "abcdbacbcadd"
    print("max substring is ",longest_substring_distinct(s)) 
