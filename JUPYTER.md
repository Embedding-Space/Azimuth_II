# JUPYTER.md

**Working with Jupyter Notebooks in Claude Code**

This document contains hard-won wisdom about editing Jupyter notebooks programmatically. These patterns prevent the most common mistakes and keep notebooks clean and correct.

---

## The Golden Rules

1. **Read before you edit** - Always use the Read tool first to understand current structure
2. **One edit at a time** - Make ONE NotebookEdit call, then stop and verify
3. **Use the right tool for the job** - NotebookEdit for notebooks, not bash/jq/Python scripts
4. **When in doubt, delete and rebuild** - It's faster than trying to fix insertion order bugs

---

## Cell Insertion: How It Actually Works

### The Anchor Point Rule

**`NotebookEdit` with `edit_mode='insert'` and `cell_id='cell-X'` inserts the new cell AFTER cell-X.**

```
Before:
  cell-5: Some code
  cell-6: Some markdown

After inserting new cell with cell_id='cell-5':
  cell-5: Some code
  cell-NEW: Your new cell    ← Inserted here (becomes the new cell-6)
  cell-6: Some markdown      ← Old cell-6 becomes cell-7
```

### Inserting Multiple Cells in Sequence

If you need to add cells A, B, C after cell-5:

**Option 1: Insert in reverse order (all using same anchor)**
```python
# Insert C after cell-5 → becomes cell-6
# Insert B after cell-5 → becomes cell-6, C becomes cell-7
# Insert A after cell-5 → becomes cell-6, B becomes cell-7, C becomes cell-8
# Result: cell-5, A, B, C, old-cell-6
```

**Option 2: Track new cell IDs and chain (preferred for clarity)**
```python
# Insert A after cell-5 → note the new cell ID returned
# Insert B after cell-A → note the new cell ID returned
# Insert C after cell-B → done
# Result: cell-5, A, B, C, old-cell-6
```

**Option 3: Do it across multiple user interactions**
- Make ONE insertion
- Let user verify it looks right
- Continue with next insertion
- This is ALWAYS the safest approach

### Common Mistakes

❌ **Don't do this:**
```
Insert cell after cell-5
Insert another cell after cell-5  ← This goes BETWEEN cell-5 and your first insertion!
```

❌ **Don't do this:**
```
Read notebook
Insert cell after cell-5
Insert cell after cell-6  ← cell-6 is now cell-7! You're inserting in the wrong place!
```

✅ **Do this instead:**
```
Read notebook
Insert cell A after cell-5
STOP - let user verify
(Next interaction) Insert cell B after cell-A
STOP - let user verify
```

---

## Cell Deletion: Fixing Mistakes

When you screw up (and you will), **use NotebookEdit with `edit_mode='delete'`**.

### The Right Way to Fix Duplicate Cells

```python
# You inserted cells wrong and now have duplicates
# DON'T try to fix with bash/jq/Python
# DO use NotebookEdit to delete the bad cells

NotebookEdit(
    notebook_path="path/to/notebook.ipynb",
    cell_id="cell-X",  # The bad cell
    edit_mode="delete",
    new_source=""  # Required but ignored for deletion
)
```

### Deletion Strategy

1. **Identify the bad cells** - Read the notebook and note which cells are wrong
2. **Delete from bottom up** - Delete higher-numbered cells first to avoid index shifting
3. **One at a time** - Delete one cell, verify, then delete next
4. **Rebuild if needed** - If structure is really messed up, delete ALL the bad cells, then insert correct ones

---

## Cell Replacement: Editing Existing Cells

Use `edit_mode='replace'` (the default) to change an existing cell's content:

```python
NotebookEdit(
    notebook_path="path/to/notebook.ipynb",
    cell_id="cell-5",
    new_source="print('New content')"
    # edit_mode='replace' is the default
)
```

This is **safe** - it doesn't change cell order, just content.

---

## Reading Notebooks: Understanding Structure

### The Read Tool Format

When you Read a notebook, you get:

```
<cell id="cell-0"><cell_type>markdown</cell_type>
# Title
Some content
</cell>

<cell id="cell-1"><cell_type>code</cell_type>
import pandas as pd
print("Hello")
</cell>
```

### Quick Structure Check

To see cell order without reading full content:

```bash
cat notebook.ipynb | jq '.cells | map({type: .cell_type, lines: (.source | length)})'
```

This shows you cell types and sizes without dumping all the content.

### Finding Specific Cells

To find the cell containing specific code:

```bash
cat notebook.ipynb | jq -r '.cells[] | select(.source | join("") | contains("search_string")) | .id'
```

But honestly, just **use the Read tool** - it's clearer and safer.

---

## Cell IDs: A Critical Detail

### How Cell IDs Work

- Jupyter notebooks may or may not have explicit cell IDs in the JSON
- Claude Code assigns IDs like `cell-0`, `cell-1`, etc. based on position
- **These IDs are positional** - if you insert/delete cells, IDs shift

### What This Means for You

**Within a single NotebookEdit operation:**
- Cell IDs are stable

**Across multiple NotebookEdit operations:**
- Cell IDs may shift
- Always re-read the notebook if you need to make additional edits
- Don't rely on cell IDs from a previous read after you've modified the notebook

---

## Common Patterns and Recipes

### Pattern 1: Adding Analysis Section Between Existing Cells

**Goal:** Insert new section between cell-5 and cell-6

```
1. Read notebook to confirm structure
2. Insert markdown header after cell-5
3. STOP - verify with user
4. Insert code cell after the new markdown header
5. STOP - verify with user
```

### Pattern 2: Fixing a Mistaken Insertion

**Goal:** You inserted cells in wrong order

```
1. Read notebook to see the mess
2. Delete the incorrectly placed cells (from highest to lowest cell number)
3. STOP - verify structure is clean
4. Re-insert cells in correct order (following insertion rules)
5. STOP - verify with user
```

### Pattern 3: Replacing a Cell's Content

**Goal:** Update existing cell without changing structure

```
1. Read notebook to confirm cell ID
2. Use NotebookEdit with edit_mode='replace'
3. Done - replacement is atomic and safe
```

---

## What NOT To Do

### ❌ Don't Use Bash/JQ/Python to Fix Notebook Structure

When you make a mistake, your instinct will be to write a Python script or bash pipeline with jq to "fix" the notebook JSON.

**Don't do this.**

Jupyter notebook JSON is complex with metadata, execution counts, outputs, etc. Hand-editing it is error-prone and creates broken notebooks.

**Instead:** Use NotebookEdit to delete bad cells and insert correct ones.

### ❌ Don't Make Multiple Insertions Without Verification

**Don't do this:**
```
Insert cell A
Insert cell B  ← You don't know where this actually went!
Insert cell C  ← Now you're really lost
Try to fix with bash ← Now you're in deep trouble
```

**Do this:**
```
Insert cell A
STOP and tell user what you did
Wait for user to verify
(Next turn) Continue with cell B
```

### ❌ Don't Trust Cell IDs After Modifications

**Don't do this:**
```
Read notebook (cell-6 is markdown)
Insert new cell after cell-5
Try to edit cell-6  ← This is now cell-7!
```

**Do this:**
```
Read notebook
Insert new cell after cell-5
Read notebook again  ← Fresh structure
Now edit the cell you actually want
```

---

## Debugging Checklist

When notebook edits go wrong, work through this checklist:

1. **Read the notebook** - What's the actual current state?
2. **Identify the problem** - Which cells are wrong/duplicated/missing?
3. **Plan the fix** - Write down the sequence of deletions and insertions needed
4. **Execute one step** - Make ONE NotebookEdit call
5. **Verify** - Read notebook again or ask user to check
6. **Repeat** - Continue with next step only after verification

**Do NOT:**
- Try to fix everything in one response
- Use bash/jq/Python to manipulate notebook JSON
- Guess at cell positions without reading first
- Make multiple edits without user verification

---

## Summary: The Safe Workflow

1. **Read first** - Always know current structure
2. **Plan explicitly** - "I will insert cell X after cell-Y because..."
3. **One edit at a time** - Single NotebookEdit call per response
4. **Stop and verify** - Let user confirm it worked
5. **Use NotebookEdit for fixes** - delete/insert, not bash scripts
6. **When lost, delete and rebuild** - Faster than debugging complex insertion errors

Follow these rules and you'll never create another tangled notebook mess.

---

*This document is included from CLAUDE.md and provides specific guidance for Jupyter notebook operations in the Azimuth project and similar research tinkering.*
