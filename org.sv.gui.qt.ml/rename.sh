#!/bin/bash

#Script to rename seg2d files to seg2d ml to prevent naming conflicts
# s="The dog"
# a="dog"
# b="cat"
# echo "${s/$a/$b}"

find ./src -name "svContourGroup*" -exec bash -c 'mv "$0" "${0/svContourGroup/svMLContourGroup}"'
find ./src -name "svLevelSet*" -exec bash -c 'mv "$0" "${0/svLevelSet/svMLLevelSet}"'
find ./src -name "svLoft*" -exec bash -c 'mv "$0" "${0/svLoft/svMLLoft}"'
find ./src -name "svSeg2D*" -exec bash -c 'mv "$0" "${0/svSeg2D/svMLSeg2D}"'
find ./src -name "svSeg3D*" -exec bash -c 'mv "$0" "${0/svSeg3D/svMLSeg3D}"'
find ./src -name "svSegmentation*" -exec bash -c 'mv "$0" "${0/svSegmentation/svMLSegmentation}"'

for f in $(find .)
do
  echo $f
  if [ $f = ./rename.sh ]
  then
    echo "file = ${f}, skipping"
  else
    sed -i 's/svContourGroup/svMLContourGroup/g' $f
    sed -i 's/svLevelSet/svMLLevelSet/g' $f
    sed -i 's/svLoft/svMLLoft/g' $f
    sed -i 's/svSeg2D/svMLSeg2D/g' $f
    sed -i 's/svSeg3D/svMLSeg3D/g' $f
    sed -i 's/svSegmentation/svMLSegmentation/g' $f

    sed -i 's/SVCONTOURGROUP/SVMLCONTOURGROUP/g' $f
    sed -i 's/SVLEVELSET/SVMLLEVELSET/g' $f
    sed -i 's/SVLOFT/SVMLLOFT/g' $f
    sed -i 's/SVSEG2D/SVMLSEG2D/g' $f
    sed -i 's/SVSEG3D/SVMLSEG3D/g' $f
    sed -i 's/SVSEGMENTATION/SVMLSEGMENTATION/g' $f
  fi
done
