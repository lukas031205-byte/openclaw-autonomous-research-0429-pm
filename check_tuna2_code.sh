#!/bin/bash
# check_tuna2_code.sh
# Checks if Tuna-2 (Meta/HKU/Waterwater) code has been released on GitHub
# Run each autonomous research window to monitor status

echo "=== Tuna-2 Code Status Check $(date) ==="
echo "URL: https://github.com/meta-llama/tuna2"
echo ""

# Check HTTP status code
echo -n "HTTP status: "
HTTP_CODE=$(curl -s --max-time 15 -o /dev/null -w "%{http_code}" https://github.com/meta-llama/tuna2)
echo "$HTTP_CODE"
echo ""

# Check for code release indicators
echo "Checking page content for code status..."
PAGE_CONTENT=$(curl -s --max-time 15 https://github.com/meta-llama/tuna2 2>/dev/null)

# Look for "code coming soon", "private", "not released" type messages
if echo "$PAGE_CONTENT" | grep -qi "code.*coming\|code.*soon\|not.*released\|private\|code.*available\|open.*source\|public"; then
    echo "Code status text found:"
    echo "$PAGE_CONTENT" | grep -i "code.*coming\|code.*soon\|not.*released\|private\|code.*available\|open.*source\|public" | head -5
else
    echo "No explicit code-release status text found"
    echo "(page may have code tab present = RELEASED)"
fi

echo ""
echo "=== Check complete ==="