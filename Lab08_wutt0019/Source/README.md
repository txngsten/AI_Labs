# COMP3742 Lab 8 Instructions - Oliver Wuttke (WUTT0019)
To start head to [SWI Prolog](https://www.swi-prolog.org/download/stable) download page and setup and install Prolog on your system.
Once that is done follow these steps to setup and run the lab.

Run this to launch the Prolog interpreter:
```bash
swipl-win
```

Then go ahead and link the [panthera](panthera.pl) knowledge base and rules with:
```swipl-win
[panthera].
```

Question 3 query:
```swipl-win
parent(Grandparent, P), parent(P, cave_lion).
```

Question 4 query:
```swipl-win
parent(ancestor4, C).
```

Question 5 query:
```swipl-win
parent(P, lion).
```