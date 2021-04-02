# Python - Basics

## Python
- High−level.
- Clean, readable, and efficient.
- Easy and fun to learn.
- Dynamic.
- Fast to write and test code.
- Less code.
- Flexible.
- Interactive.
- Great support
- Open source.
- Vast range of libraries.
- Huge number of users.
- [Slow (sometimes)](/tips_to_speed_up_python).

## Data types

x = 10
type(x)

y = 0.5
type(y)

x + y

z = x / 2
z

type(z)

int(z)

my_string = 'Hello'
type(my_string)

print(my_string)

my_string.upper()

greeting = 'Hello, I am Gary.'
greeting

greeting.split(' ')

greeting.replace("Gary", "Gary's Mother")

'-'.join(greeting)

my_list = [1, 2, 3, 4, 5]
type(my_list)

len(my_list)

my_list[0]

my_list[-1]

my_list[-1] = my_string
my_list

my_list.append(x)

my_list

[num for num in range(10)]

my_dict = {'a': 1, 'b': 2, 'c': 3}
type(my_dict)

my_dict.keys()

my_dict.values()

my_dict['b']

my_dict.update({'d': greeting})

my_dict

my_boolean = True
type(my_boolean)

my_boolean == True

my_boolean == False

## Control flow

if 10 < 20:
    print(f'Truthy')

if True:
    print(f'Truthy')

for num in range(5):
    print(num)

for num in range(5):
    if num % 2 == 0:
        print(f'{num} is even')

for num in range(5):
    if num % 2 == 0:
        print(f'{num} is even')
    else:
        print(f'{num} is odd')

colours = ['red', 'green', 'blue']
for index, colour in enumerate(colours):
    print(index, colour)

my_dict = {'a': 1, 'b': 2, 'c': 3}

for key, value in my_dict.items():
    print(key, value)

## Modules

import os

os.getcwd()

import glob

glob.glob('*bash')

import re

re.findall(r'\d+', 'Hello 12345')

## Functions

def say_hello(*args): # arguments
    for arg in args:
        print(f'Hello {arg}!')

say_hello('Gary')

say_hello('Gary', "Gary's Mother")

def do_maths(x=2, y=3, z=4): # keyword arguments
    print(f'x = {x}')
    print(f'y = {y}')
    print(f'z = {z}')
    result = (x + y) * z
    print(f'result = {result}')

do_maths()

do_maths(1, 2, 3)

do_maths(3, 2, 1)

do_maths(z=3, y=2, x=1)

def do_maths(**kwargs): # keyword arguments
    print(kwargs)
    return sum(kwargs.values())

result = do_maths(x=2, y=3, z=4)

result

For more information, see the [documentation](https://docs.python.org/3/).

