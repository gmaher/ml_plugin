set(SRC_CPP_FILES

)

set(INTERNAL_CPP_FILES
    svMLLoftingUtils.cxx
    svMLContourGroupCreate.cxx
    svMLContourGroupCreateAction.cxx
    svMLSegmentationLegacyLoadAction.cxx
    svMLSegmentationLegacySaveAction.cxx
    svMLSegmentationLoadAction.cxx
    svMLLevelSet2DWidget.cxx
    svMLLoftParamWidget.cxx
    svMLSeg2DEdit.cxx
    svMLContourGroupPoint2DSizeAction.cxx
    svMLContourGroupPoint3DSizeAction.cxx
    svMLSeg3DCreateAction.cxx
    svMLSeg3DEdit.cxx
    svMLSegmentationPluginActivator.cxx
)

set(MOC_H_FILES
    src/internal/svMLLoftingUtils.h
    src/internal/svMLContourGroupCreate.h
    src/internal/svMLContourGroupCreateAction.h
    src/internal/svMLSegmentationLegacyLoadAction.h
    src/internal/svMLSegmentationLegacySaveAction.h
    src/internal/svMLSegmentationLoadAction.h
    src/internal/svMLLevelSet2DWidget.h
    src/internal/svMLLoftParamWidget.h
    src/internal/svMLSeg2DEdit.h
    src/internal/svMLContourGroupPoint2DSizeAction.h
    src/internal/svMLContourGroupPoint3DSizeAction.h
    src/internal/svMLSeg3DCreateAction.h
    src/internal/svMLSeg3DEdit.h
    src/internal/svMLSegmentationPluginActivator.h
)

set(UI_FILES
    src/internal/svMLContourGroupCreate.ui
    src/internal/svMLLevelSet2DWidget.ui
    src/internal/svMLLoftParamWidget.ui
    src/internal/svMLSeg2DEdit.ui
    src/internal/svMLSeg3DEdit.ui
)

set(CACHED_RESOURCE_FILES
  plugin.xml
  resources/contourgroup.png
  resources/svseg3d.png
)

set(QRC_FILES

)

set(CPP_FILES )

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})

