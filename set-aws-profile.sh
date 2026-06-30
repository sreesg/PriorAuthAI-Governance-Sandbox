#!/bin/zsh
# ============================================================================
# set-aws-profile.sh — Set AWS credentials for CRF deployment
#
# USAGE (pass as arguments — no interactive prompts):
#
#   source ./set-aws-profile.sh <ACCESS_KEY_ID> <SECRET_ACCESS_KEY> [SESSION_TOKEN] [REGION]
#
# EXAMPLES:
#   source ./set-aws-profile.sh AKIAIOSFODNN7EXAMPLE wJalrXUtnFEMI/K7MDENG/bPxRfi
#   source ./set-aws-profile.sh AKIAXXXX secret_key session_token us-west-2
#
# ============================================================================

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: source ./set-aws-profile.sh <ACCESS_KEY_ID> <SECRET_ACCESS_KEY> [SESSION_TOKEN] [REGION]"
    echo ""
    echo "Example:"
    echo "  source ./set-aws-profile.sh AKIA... MySecretKey... IQoJb3... us-east-1"
    return 1 2>/dev/null || exit 1
fi

unset AWS_PROFILE

export AWS_ACCESS_KEY_ID="$1"
export AWS_SECRET_ACCESS_KEY="$2"
export AWS_DEFAULT_REGION="${4:-us-east-1}"

if [ -n "$3" ]; then
    export AWS_SESSION_TOKEN="$3"
fi

echo "✓ AWS credentials set"
echo "  Region: $AWS_DEFAULT_REGION"
echo "  Key:    ${1:0:8}..."
echo "  Active in this terminal session only."
