# Prompt Instruction Recipes

Is AI ignore

## General

```markdown
# First time Junie run 
Study the guidelines at `.junie/guidelines.md` thoroughly.
Provide a comprehensive summary of all guidelines studied.

# Evaluate codebase
Given your study of the guidelines at `.junie/guidelines.md`, analyse the super project and make recommendations if relevant.   
```

```markdown
# Tasks
1. Analyze codebase thoroughly to understand what it is about.
2. TODO

# Instructions
- Always comply with `.junie/guidelines.md` guidelines.
- Execute all tests before submitting.
```

```markdown
Read and implement the plan at `.junie/active_plans/TODO.md`.
```

```markdown
# Tasks
1. Analyze codebase thoroughly to understand what it is about.
2. Implement script/function `NAME` in `src/lib/commands/NAME.bash` with the following features:
   - logic for blablabla; 
   - implement a `-F|--FLAG` flag to blablabla;
   - default case: 
     - blablab. 
   Inspire yourself with `src/lib/commands/OTHER_NAME.bash` for the implementation.
   Implement bats tests for `src/lib/commands/NAME.bash`.
   Create at least one test case per cli options.
   Inspire yourself with `tests/tests_bats/test_OTHER_NAME.bats`.
   Implement Markdown documentation at `documentation/command/NAME.md`.
3. Check if any Markdown documentation at `documentation/` need to be updated.

# Instructions
- Always comply with `.junie/guidelines.md` guidelines.
- Execute all unit-tests and all integration tests before submitting.
```
