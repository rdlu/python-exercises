#!/bin/python3

import math
import os
import random
import re
import sys
from collections import Counter

# Complete the repeatedString function below.
# s: a string to repeat 
# n: the number of characters to consider 
# aba, 10 returns 7
def repeatedString(s, n):
    # return (s * n)[:n].count('a') -> memory error
    # return (s * (n // len(s)))[:n].count('a') -> memory error
    # "a" count of full string * number of string repeats + "a" count of last remnant string.
    return s.count("a") * (n // len(s)) + s[:n % len(s)].count("a")

# Complete the sockMerchant function below.
# n: the number of socks in the pile
# ar: the colors of each sock
# 9, [10 20 20 10 10 30 50 10 20] returns 3
def sockMerchant(n, ar):
    return sum([math.floor(values/2) for values in Counter(ar).values()])

# Complete the rotLeft function below.
def rotLeft(a, d):
    # r = d % len(a) # optimizing number of slice rotations
    return a[d:]+a[:d]

def right_rotation(a, k):
   # if the size of k > len(a), rotate only necessary with
   # module of the division
   rotations = k % len(a)
   return a[-rotations:] + a[:-rotations]

# hourglassSum
def hourglassSum(arr):
    return max(
        [sum(arr[i-1][j-1:j+2] +    \
                [arr[i][j]]    +    \
                arr[i+1][j-1:j+2])  \
            for j in range(1, 5)    \
            for i in range(1, 5)])

# arr = unordered range(1,n), without gaps and starting 1
def minimumSwaps(arr):
    numSwaps = 0
    i = 0
    while(i < len(arr)-1):
        if arr[i] != i+1:
            tmp = arr[i]
            arr[i], arr[tmp-1] = arr[tmp-1], arr[i]
            numSwaps += 1
        else:
            i += 1
    return numSwaps
#   a b k    n = 10
#    1 5 3
#    4 8 7
#    6 9 1
# 0: [0] * 10
# 1: 
def arrayManipulation(n, queries):
    # arr = [0 for i in range(n)]
    # O(n2)
    arr = [0] * n
    for start, end, k in queries:
        for i in range(start-1,end):
            arr[i] += k
    return max(arr)

# O(n+1): custom array of differences i-1,i
def arrayManipulation2(n, queries):
    my_array = [0] * (n+1)
    count = 0
    res = 0
    for first,last,value in queries:
        my_array[first-1] += value
        my_array[last] -= value
    
    for item in my_array:
        count += item
        if count > res:
            res = count      
    return res

# How many "bribes" people used to get in first position in the queue
# If person bribed more than 2, its too chaotic
# 2 .1 5 .3 .4 => 3
# 2 5 1 3 4 => Too chaotic
def minimumBribes(queue):
    n = 0 
    # align indices to zero
    queue = [person-1 for person in queue]
 
    for position, person in enumerate(queue):
        if person - position > 2:
            return "Too chaotic"
        # 1 .0 4 .2 .3
        for closer_position in range(max(person-1,0),position):
            if queue[closer_position] > person:
                n += 1
    return n
