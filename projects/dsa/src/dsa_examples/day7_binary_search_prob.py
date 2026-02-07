def binary_search(arr, num):
     left = 0
     right = len(arr) -1 

     while (left<=right):
         mid = (left+right)//2
         if num == arr[mid]:
             return mid
         elif num >  arr[mid]:
             left = mid+1
         else:
            right = mid-1
     return -1


def lower_bound(arr, num):
    lower = 0
    upper = len(arr) 
    while (lower<upper):
        mid = ( lower + upper ) //2
        if arr[mid] < num :
            lower = mid+1
        else:
            upper = mid
    return lower

def upper_bound(arr, num):
    lower = 0
    upper = len(arr) 
    while(lower<upper):
        mid = ( lower+upper ) //2
        if arr[mid] <= num :
            lower = mid +1 
        else:
            upper = mid
    return lower




if __name__ == "__main__":
    arr = [3,4,2,2,7,8]
    print("lower bound is ", lower_bound(arr, 2))
   # print("then index of num is ", binary_search(arr,9))
