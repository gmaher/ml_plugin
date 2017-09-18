#!/bin/bash

#Script to rename seg2d files to seg2d ml to prevent naming conflicts
# s="The dog"
# a="dog"
# b="cat"
# echo "${s/$a/$b}"

find ./src -name "svContourGroupCreate*" -exec bash -c 'mv "$0" "${0/svContourGroupCreate/svMLContourGroupCreate}"' {} \;
find ./src -name "svContourGroupPoint*" -exec bash -c 'mv "$0" "${0/svContourGroupPoint/svMLContourGroupPoint}"' {} \;
find ./src -name "svLevelSet*" -exec bash -c 'mv "$0" "${0/svLevelSet/svMLLevelSet}"' {} \;
find ./src -name "svLoft*" -exec bash -c 'mv "$0" "${0/svLoft/svMLLoft}"' {} \;
find ./src -name "svSeg2D*" -exec bash -c 'mv "$0" "${0/svSeg2D/svMLSeg2D}"' {} \;
find ./src -name "svSeg3D*" -exec bash -c 'mv "$0" "${0/svSeg3D/svMLSeg3D}"' {} \;
find ./src -name "svSegmentationLegacyLoad*" -exec bash -c 'mv "$0" "${0/svSegmentationLegacyLoad/svMLSegmentationLegacyLoad}"' {} \;
find ./src -name "svSegmentationLoad*" -exec bash -c 'mv "$0" "${0/svSegmentationLoad/svMLSegmentationLoad}"' {} \;
find ./src -name "svSegmentationLegacySave*" -exec bash -c 'mv "$0" "${0/svSegmentationLegacySave/svMLSegmentationLegacySave}"' {} \;
find ./src -name "svSegmentationPlugin*" -exec bash -c 'mv "$0" "${0/svSegmentationPlugin/svMLSegmentationPlugin}"' {} \;


for f in $(find .)
do
  echo $f
  if [ $f = ./rename.sh ]
  then
    echo "file = ${f}, skipping"
  else
    sed -i 's/svContourGroupCreate/svMLContourGroupCreate/g' $f
    sed -i 's/svContourGroupPoint/svMLContourGroupPoint/g' $f
    sed -i 's/svLevelSet/svMLLevelSet/g' $f
    sed -i 's/svLoft/svMLLoft/g' $f
    sed -i 's/svSeg2D/svMLSeg2D/g' $f
    sed -i 's/svSeg3D/svMLSeg3D/g' $f
    sed -i 's/svSegmentationLegacyLoad/svMLSegmentationLegacyLoad/g' $f
    sed -i 's/svSegmentationLegacySave/svMLSegmentationLegacySave/g' $f
    sed -i 's/svSegmentationLoad/svMLSegmentationLoad/g' $f
    sed -i 's/svSegmentationPlugin/svMLSegmentationPlugin/g' $f


    sed -i 's/SVCONTOURGROUPCREATE/SVMLCONTOURGROUPCREATE/g' $f
    sed -i 's/SVCONTOURGROUPPOINT/SVMLCONTOURGROUPPOINT/g' $f
    sed -i 's/SVLEVELSET/SVMLLEVELSET/g' $f
    sed -i 's/SVLOFT/SVMLLOFT/g' $f
    sed -i 's/SVSEG2D/SVMLSEG2D/g' $f
    sed -i 's/SVSEG3D/SVMLSEG3D/g' $f
    sed -i 's/SVSEGMENTATIONLEGACYLOAD/SVMLSEGMENTATIONLEGACYLOAD/g' $f
    sed -i 's/SVSEGMENTATIONLEGACYSAVE/SVMLSEGMENTATIONLEGACYSAVE/g' $f
    sed -i 's/SVSEGMENTATIONLOAD/SVMLSEGMENTATIONLOAD/g' $f
    sed -i 's/SVSEGMENTATIONPLUGIN/SVMLSEGMENTATIONPLUGIN/g' $f

  fi
done
