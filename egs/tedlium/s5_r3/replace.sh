
set -e

for f in $(grep -rl "/home/rsw2148_columbia_edu/kaldi" .); do
  echo "replacing in $f"
  sed -i "s/\/home\/zy2388\/kaldi-trunk/\/home\/rsw2148_columbia_edu\/kaldi/g" $f
done
