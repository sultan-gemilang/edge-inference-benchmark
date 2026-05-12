#!/bin/bash

# This script downloads the necessary files for the Qwen-Edge benchmark tests.
# Replace ACCESS_TOKEN, FILE_ID, and FILE_NAME with the appropriate values.
ACCESS_TOKEN=""
FILE_ID1="1e_KysaEV66RX7psiE3-hm0Yxf4qK6EKt"
FILE_NAME1="qwen2_attn.tar.gz"

curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/$FILE_ID1?alt=media -o $FILE_NAME1