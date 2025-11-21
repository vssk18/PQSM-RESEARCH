#!/bin/bash

# PQSM RESEARCH - COMPLETE TERMINAL COMMANDS
# Run these commands in order to fix everything

echo "================================================="
echo "PQSM RESEARCH - COMPLETE FIX TERMINAL COMMANDS"
echo "================================================="
echo ""

# STEP 1: Download the corrected package
echo "STEP 1: Download and extract corrected package"
echo "-----------------------------------------------"
cat << 'EOF'
cd ~/Downloads
wget https://github.com/vssk18/PQSM-FINAL-CORRECTED.zip
unzip PQSM-FINAL-CORRECTED.zip
cd PQSM-FINAL-CORRECTED
EOF

echo ""
echo "STEP 2: Copy to your existing repository"
echo "----------------------------------------"
cat << 'EOF'
cd ~/PQSM-RESEARCH
cp -r ~/Downloads/PQSM-FINAL-CORRECTED/* .
EOF

echo ""
echo "STEP 3: Run the automatic fix script"
echo "------------------------------------"
cat << 'EOF'
chmod +x fix_everything.sh
./fix_everything.sh
EOF

echo ""
echo "STEP 4: Verify the fixes"
echo "------------------------"
cat << 'EOF'
# Check README has correct numbers
grep "4,608" README.md

# Check data file
wc -l analysis/analysis/all_runs_merged.csv

# Check plots exist
ls -la analysis/analysis/plots/*.png | wc -l
ls -la analysis/analysis/advanced_plots/*.png | wc -l
EOF

echo ""
echo "STEP 5: Commit and push to GitHub"
echo "---------------------------------"
cat << 'EOF'
git add -A
git commit -m "Critical Fix: Document actual 4,608 experimental runs, fix all inconsistencies"
git push origin main
EOF

echo ""
echo "STEP 6: Compile LaTeX paper (optional)"
echo "--------------------------------------"
cat << 'EOF'
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cd ..
EOF

echo ""
echo "================================================="
echo "ALTERNATIVE: One-line fix everything"
echo "================================================="
cat << 'EOF'
curl -L https://raw.githubusercontent.com/vssk18/PQSM-RESEARCH/main/fix_everything.sh | bash
EOF

echo ""
echo "================================================="
echo "WHAT THIS FIXES:"
echo "================================================="
echo "✓ Updates README to show actual 4,608 runs (not 12,600)"
echo "✓ Fixes all LaTeX tables to show 768 runs per KEM"
echo "✓ Updates grid to show 4 latencies × 4 loss rates"
echo "✓ Fixes LaTeX compilation issues (removes dblfloatfix)"
echo "✓ Adds proper table spacing"
echo "✓ Ensures consistency across all files"
echo ""
echo "Your research is SOLID with 4,608 runs!"
echo "Don't claim more than you have - be honest!"
