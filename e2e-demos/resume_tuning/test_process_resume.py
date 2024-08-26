
import os
import re

with open("resum_out.txt", "r") as f:
    lines=f.read().replace("\\n", '\n').splitlines()
    for l in lines:
        print(l)
        print("-----------------")