largest_so_far = -1
print (f"Before: {largest_so_far}")
for num in [9,41,12,3,74,15]:
    if num > largest_so_far:
        largest_so_far = num
        print(largest_so_far, num)

print (f"After: {largest_so_far}")

count = 0
sum = 0
for i in [9,41,12,3,74,15]:
    count = count + 1
    sum = sum + i
print ('Average', int(sum/count))