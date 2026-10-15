#!/usr/bin/env python3
"""Print the rows of a recipe table as ';'-separated fields for the shell driver:
scenario_id;mode;schedule;sector;seed;route;profile;covtype;replicate"""
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    print(";".join([r["scenario_id"], r["mode"], r["schedule"], r["sector"] or "-", r["seed"],
                    r["route"], r["profile"] or "std", r["covtype"] or "fixed", r["replicate"] or "0"]))
