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

########################################################################################


def min_eating_speed(piles, h):
    low = 0
    high = max(piles)
    while low < high:
        mid = (low+high)//2
        if can_finish(mid,piles,h):
            high = mid
        else:
            low = mid+1
    return low

def can_finish(k,piles,H):
    hours = 0
    for pile in piles:
        hours += (pile+k-1)//k
    return hours <= H
####################################################################


if __name__ == "__main__":
    #arr = [3,4,2,2,7,8]
    #print("lower bound is ", lower_bound(arr, 2))
   # print("then index of num is ", binary_search(arr,9))
   H = 20
   piles = [392, 493, 768,690,350]
   print("min speed is " , min_eating_speed(piles,H))

