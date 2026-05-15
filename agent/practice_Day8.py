smallest = None
print (f"Before: {smallest}")

for num in [9,41,12,3,74,15]:
    if smallest is None:
        smallest = num
    elif num < smallest:
        smallest = num
        print(smallest)

print (f"After: {smallest}")