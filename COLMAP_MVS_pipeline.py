#!/usr/bin/python

import argparse, os, subprocess, time, math, sys, errno, logging

MVSDirectory = ""
outputDirectory = ""


def createParser():
    parser = argparse.ArgumentParser(description="OpenMVG/OpenMVS pipeline")
    parser._action_groups.pop()

    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--input", type=str, help="Input images folder", required=True
    )
    required.add_argument("--output", type=str, help="Output path", required=True)

    required.add_argument(
        "--sfm-type",
        type=str,
        help="Select SfM type: global or incremental",
        choices=["global", "incremental", "incrementalv2"],
        required=True,
    )

    pipelines = parser.add_argument_group("Pipelines to run (min. 1 required)")
    pipelines.add_argument(
        "--run-openmvg", action="store_true", help="Run OpenMVG pipeline"
    )
    pipelines.add_argument(
        "--run-openmvs", action="store_true", help="Run OpenMVS pipeline"
    )
    pipelines.add_argument(
        "--run-colmap", action="store_true", help="Run Colmap pipeline"
    )

    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--debug", action="store_true", help="Print commands without executing them"
    )
    optional.add_argument(
        "--recompute", action="store_true", help="Recompute everything"
    )
    optional.add_argument(
        "--openmvg", type=str, help="Location of openmvg. Default: /opt/openmvg"
    )
    optional.add_argument(
        "--openmvs", type=str, help="Location of openmvs. Default: /opt/openmvs"
    )
    optional.add_argument(
        "--colmap", type=str, help="Location of colmap. Default: /opt/colmap"
    )
    optional.add_argument(
        "--use_gpu",
        type=bool,
        choices=[0, 1],
        help="whether use gpu for feature extractor",
    )

    openmvg = parser.add_argument_group("OpenMVG")
    openmvg.add_argument(
        "--colorize", action="store_true", help="Create colorized sparse pointcloud"
    )

    # echo ">>>>>>>>>>>>>>Starting colmap feature extraction"
    # colmap feature_extractor \
    # --SiftExtraction.use_gpu $use_gpu \
    # --ImageReader.camera_model OPENCV \
    # --database_path $database_folder \
    # --image_path $images_folder \
    colmap_feature_extractor = parser.add_argument_group("Colmap feature extractor")
    colmap_feature_extractor.add_argument(
        "--camera_model",
        type=str,
        help="select camera model",
        choices=["SIMPLE_RADIAL", "OPENCV"],
    )

    # colmap exhaustive_matcher \
    # --SiftMatching.use_gpu $use_gpu \
    # --database_path $database_folder \
    colmap_exhaustive_matcher = parser.add_argument_group("Colmap exhaustive matcher")

    # colmap mapper \
    # --database_path $database_folder \
    # --image_path $images_folder \
    # --output_path $output_folder \
    colmap_mapper = parser.add_argument_group("Colmap mapper")

    # colmap image_undistorter \
    # --image_path $images_folder \
    # --input_path $output_folder/0 \
    # --output_path $working_folder/dense \
    # --output_type COLMAP \
    colmap_image_undistorter = parser.add_argument_group("Colmap image undistorter")
    colmap_image_undistorter.add_argument(
        "--output_type",
        type=str,
        choices=["COLMAP"],
        help="select image undistorter output type",
    )
    # colmap model_converter \
    # --input_path $working_folder/dense/sparse \
    # --output_path $working_folder/dense/sparse \
    # --output_type TXT
    colmap_model_converter = parser.add_argument_group("Colmap model converter")

    # sudo ./InterfaceCOLMAP \
    # --working-folder $working_folder \
    # -i $working_folder/dense/ \
    # --output-file $working_folder/model_colmap.mvs

    imageListing = parser.add_argument_group("OpenMVG Image Listing")
    imageListing.add_argument(
        "--cgroup",
        action="store_true",
        help="Each view has it's own camera intrisic parameters",
    )
    imageListing.add_argument(
        "--flength",
        type=float,
        help="If your camera is not listed in the camera sensor database, you can set pixel focal length here. The value can be calculated by max(width-pixels, height-pixels) * focal length(mm) / Sensor width",
    )
    imageListing.add_argument(
        "--cmodel",
        type=int,
        help="Camera model: 1. Pinhole 2. Pinhole Radial 1 3. Pinhole Radial 3 (Default) 4. Pinhole Brown 5. Pinhole with a Simple Fish-eye Distortion",
        choices=[1, 2, 3, 4, 5],
    )

    computeFeature = parser.add_argument_group("OpenMVG Compute Features")
    computeFeature.add_argument(
        "--descmethod",
        type=str,
        help="Method to describe and image. Default: SIFT",
        choices=["SIFT", "AKAZE_FLOAT", "AKAZE_MLDB"],
    )
    computeFeature.add_argument(
        "--dpreset",
        type=str,
        help="Used to control the Image_describer configuration. Default: NORMAL",
        choices=["NORMAL", "HIGH", "ULTRA"],
    )
    computeFeature.add_argument(
        "--upright",
        action="store_true",
        help="Use upright feature or not. 0 (default) 1: Extract upright feature",
    )

    computeMatches = parser.add_argument_group("OpenMVG Compute Matches")
    computeMatches.add_argument(
        "--ratio",
        type=float,
        help="Nearest Neighbor distance ratio (smaller is more restrictive => Less false positives). Default: 0.8",
    )
    computeMatches.add_argument(
        "--geomodel",
        type=str,
        help="Compute Matches geometric model: f: Fundamental matrix filtering (default) For Incremental SfM e: Essential matrix filtering For Global SfM h: Homography matrix filtering For datasets that have same point of projection",
        choices=["f", "e", "h"],
    )
    computeMatches.add_argument(
        "--matching",
        type=str,
        help="Compute matches nearest matching method. Default: FASTCASCADEHASHINGL2",
        choices=["BRUTEFORCEL2", "ANNL2", "CASCADEHASHINGL2", "FASTCASCADEHASHINGL2"],
    )

    incrementalSfm = parser.add_argument_group("OpenMVG Incremental SfM")
    incrementalSfm.add_argument(
        "--icmodel",
        type=int,
        help="The camera model type that will be used for views with unknown intrinsic: 1. Pinhole 2. Pinhole radial 1 3. Pinhole radial 3 (default) 4. Pinhole radial 3 + tangential 2 5. Pinhole fisheye",
        choices=[1, 2, 3, 4, 5],
    )

    globalSfm = parser.add_argument_group("OpenMVG Global SfM")
    globalSfm.add_argument(
        "--grotavg",
        type=int,
        help="1. L1 rotation averaging [Chatterjee] 2. L2 rotation averaging [Martinec] (default)",
        choices=[1, 2],
    )
    globalSfm.add_argument(
        "--gtransavg",
        type=int,
        help="1: L1 translation averaging [GlobalACSfM] 2: L2 translation averaging [Kyle2014] 3: SoftL1 minimization [GlobalACSfM] (default)",
        choices=[1, 2, 3],
    )

    openmvs = parser.add_argument_group("OpenMVS")
    openmvs.add_argument(
        "--output-obj",
        action="store_true",
        help="Output mesh files as obj instead of ply",
    )

    openmvsDensify = parser.add_argument_group("OpenMVS DensifyPointCloud")
    openmvsDensify.add_argument(
        "--densify", action="store_true", help="Enable dense reconstruction"
    )
    openmvsDensify.add_argument(
        "--densify-only", action="store_true", help="Densify pointcloud and exit"
    )
    openmvsDensify.add_argument(
        "--dnumviews",
        type=int,
        help="Number of view used for depth-map estimation. 0 for all neighbor views available. Default: 4",
    )
    openmvsDensify.add_argument(
        "--dnumviewsfuse",
        type=int,
        help="Minimum number of images that agrees with an estimate during fusion in order to consider it inliner. Default: 3",
    )
    openmvsDensify.add_argument(
        "--dreslevel",
        type=int,
        help="How many times to scale down the images before point cloud computation. For better accuracy/speed with high resolution images use 2 or even 3. Default: 1",
    )

    openmvsReconstruct = parser.add_argument_group("OpenMVS Reconstruct Mesh")
    openmvsReconstruct.add_argument(
        "--rcthickness", type=int, help="ReconstructMesh thickness factor. Default: 2"
    )
    openmvsReconstruct.add_argument(
        "--rcdistance",
        type=int,
        help="Minimum distance in pixels between the projection of two 3D points to consider them different while triangulating (0 to disable). Use to reduce amount of memory used with a penalty of lost detail. Default: 2",
    )

    openmvsRefinemesh = parser.add_argument_group("OpenMVS Refine Mesh")
    openmvsRefinemesh.add_argument(
        "--rmiterations", type=int, help="Number of RefineMesh iterations. Default: 3"
    )
    openmvsRefinemesh.add_argument(
        "--rmlevel",
        type=int,
        help="Times to scale down the images before mesh refinement. Default: 0",
    )
    openmvsRefinemesh.add_argument(
        "--rmcuda",
        action="store_true",
        help="Refine using CUDA version of RefineMesh (if available)",
    )

    openmvsTexture = parser.add_argument_group("OpenMVS Texture Mesh")
    openmvsTexture.add_argument(
        "--txemptycolor",
        type=int,
        default=0,
        help="Color of surfaces OpenMVS TextureMesh is unable to texture. Default: 0 (black)",
    )
    openmvsTexture.add_argument(
        "--txreslevel",
        type=int,
        default=0,
        help="Times to scale down the images before refiment",
    )
    return parser


def createCommands(args):
    imageListingOptions = []
    computeFeaturesOptions = []
    computePairsOptions = []
    computeMatchesOptions = []
    geometricFilterOptions = []
    incrementalSFMOptions = []
    globalSFMOptions = []
    densifyPointCloudOptions = []
    reconstructMeshOptions = []
    refineMeshOptions = []
    textureMeshOptions = []
    commands = []

    colmap_feature_extractor_options = []
    colmap_exhaustive_matcher_options = []
    colmap_mapper_options = []
    colmap_image_undistorter_options = []
    colmap_model_converter_options = []
    openmvs_InterfaceCOLMAP_options = []

    inputDirectory = args.input
    if not os.path.isabs(inputDirectory):
        inputDirectory = os.path.join(os.path.abspath("."), inputDirectory)

    global outputDirectory
    outputDirectory = args.output
    if not os.path.isabs(outputDirectory):
        outputDirectory = os.path.join(os.path.abspath("."), outputDirectory)
    matchesDirectory = os.path.join(outputDirectory, "matches")
    reconstructionDirectory = os.path.join(outputDirectory, "reconstruction_global")
    global MVSDirectory
    MVSDirectory = os.path.join(outputDirectory, "omvs")
    openmvgBin = "/opt/openmvg/bin"
    cameraSensorsDB = "/opt/openmvg/share/openMVG/sensor_width_camera_database.txt"
    openmvsBin = "/opt/openmvs/bin/OpenMVS"
    colmapBin = "/opt/colmap/bin"

    colmap_working_folder = inputDirectory
    colmap_images_folder = os.path.join(colmap_working_folder, "images")
    colmap_database_folder = os.path.join(colmap_working_folder, "database.db")
    colmap_output_folder = os.path.join(colmap_working_folder, "sparse")

    if args.openmvg != None:
        openmvgBin = os.path.join(args.openmvg, "bin")
        cameraSensorsDB = os.path.join(
            args.openmvg, "share", "openMVG", "sensor_width_camera_database.txt"
        )

    if args.openmvs != None:
        openmvsBin = os.path.join(args.openmvs, "bin", "OpenMVS")

    if args.colmap != None:
        colmapBin = os.path.join(args.colmap, "bin")

    if args.use_gpu == True:
        # print("use_gpu = " + args.use_gpu)
        use_gpu = "1"
        colmap_feature_extractor_options += ["--SiftExtraction.use_gpu", "1"]
        colmap_exhaustive_matcher_options += ["--SiftMatching.use_gpu", "1"]
        densifyPointCloudOptions += ["--cuda-device", "-1"]
        reconstructMeshOptions += ["--cuda-device", "-1"]
        refineMeshOptions += ["--cuda-device", "-1"]
        textureMeshOptions += ["--cuda-device", "-1"]
    else:
        use_gpu = "0"

    # colmap_feature_extractor
    # if args.

    # Recompute
    if args.recompute:
        computeFeaturesOptions += ["-f", "1"]
        computeMatchesOptions += ["-f", "1"]

    # OpenMVG SfM Pipeline Type
    pipelineType = args.sfm_type

    # OpenMVG Image Listing
    if args.cgroup:
        imageListingOptions += ["-g", "0"]
    if args.flength != None:
        imageListingOptions += ["-f", args.flength]
    if args.cmodel != None:
        imageListingOptions += ["-c", args.cmodel]

    # OpenMVG Compute Features
    if args.descmethod != None:
        computeFeaturesOptions += ["-m", args.descmethod.upper()]
    if args.dpreset != None:
        computeFeaturesOptions += ["-p", args.dpreset.upper()]
    if args.upright:
        computeFeaturesOptions += ["-u", "1"]

    # Geometric filter options
    if args.geomodel != None:
        geometricFilterOptions += ["-g", args.geomodel]

    # OpenMVG Match Matches
    if args.ratio != None:
        computeMatchesOptions += ["-r", args.ratio]
    if args.matching != None:
        computeMatchesOptions += ["-n", args.matching]

    # OpenMVG Inremental SfM
    if args.icmodel != None:
        incrementalSFMOptions += ["-c", args.icmodel]

    # OpenMVG Global SfM
    if args.grotavg != None:
        globalSFMOptions += ["-r", args.grotavg]
    if args.gtransavg != None:
        globalSFMOptions += ["-t", args.gtransavg]

    # OpenMVS Output Format
    openmvsOutputFormat = []
    if args.output_obj:
        openmvsOutputFormat = ["--export-type", "obj"]

    # OpenMVS Densify Mesh
    if args.dnumviewsfuse != None:
        densifyPointCloudOptions += ["--number-views-fuse", args.dnfviews]
    if args.dnumviews != None:
        densifyPointCloudOptions += ["--number-views", args.dnviews]
    if args.dreslevel != None:
        densifyPointCloudOptions += ["--resolution-level", args.dreslevel]

    # OpenMVS Reconstruct Mesh
    if args.rcthickness != None:
        reconstructMeshOptions += ["--thickness-factor", args.rcthickness]
    if args.rcdistance != None:
        reconstructMeshOptions += ["--min-point-distance", args.rcdistance]
    reconstructMeshOptions += openmvsOutputFormat

    # OpenMVS Refine Mesh
    if args.rmiterations != None:
        refineMeshOptions += ["--scales", args.rmiterations]
    if args.rmlevel != None:
        refineMeshOptions += ["--resolution-level", args.rmlevel]
    refineMeshOptions += openmvsOutputFormat

    # OpenMVS Texture Mesh
    if args.txemptycolor != None:
        textureMeshOptions += ["--empty-color", args.txemptycolor]
    if args.txreslevel != None:
        textureMeshOptions += ["--resolution-level", args.txreslevel]
    textureMeshOptions += openmvsOutputFormat

    # Create commands
    if args.run_openmvg:
        commands.append(
            {
                "title": "Instrics analysis",
                "command": [
                    os.path.join(openmvgBin, "openMVG_main_SfMInit_ImageListing"),
                    "-i",
                    inputDirectory,
                    "-o",
                    matchesDirectory,
                    "-d",
                    cameraSensorsDB,
                ]
                + imageListingOptions,
            }
        )

        commands.append(
            {
                "title": "Compute features",
                "command": [
                    os.path.join(openmvgBin, "openMVG_main_ComputeFeatures"),
                    "-i",
                    os.path.join(matchesDirectory, "sfm_data.json"),
                    "-o",
                    matchesDirectory,
                    "-m",
                    "SIFT",
                ]
                + computeFeaturesOptions,
            }
        )

        commands.append(
            {
                "title": "Compute matching pairs",
                "command": [
                    os.path.join(openmvgBin, "openMVG_main_PairGenerator"),
                    "-i",
                    os.path.join(matchesDirectory, "sfm_data.json"),
                    "-o",
                    os.path.join(matchesDirectory, "pairs.bin"),
                ],
            }
        )

        commands.append(
            {
                "title": "Compute matches",
                "command": [
                    os.path.join(openmvgBin, "openMVG_main_ComputeMatches"),
                    "-i",
                    os.path.join(matchesDirectory, "sfm_data.json"),
                    "-p",
                    os.path.join(matchesDirectory, "pairs.bin"),
                    "-o",
                    os.path.join(matchesDirectory, "matches.putative.bin"),
                ]
                + computeMatchesOptions,
            }
        )

        commands.append(
            {
                "title": "Filter matches",
                "command": [
                    os.path.join(openmvgBin, "openMVG_main_GeometricFilter"),
                    "-i",
                    os.path.join(matchesDirectory, "sfm_data.json"),
                    "-m",
                    os.path.join(matchesDirectory, "matches.putative.bin"),
                    "-o",
                    os.path.join(matchesDirectory, "matches.f.bin"),
                ]
                + geometricFilterOptions,
            }
        )

        # Select pipeline type
        if pipelineType == "global":
            commands.append(
                {
                    "title": "Do Global reconstruction",
                    "command": [
                        os.path.join(openmvgBin, "openMVG_main_SfM"),
                        "-s",
                        "GLOBAL",
                        "-i",
                        os.path.join(matchesDirectory, "sfm_data.json"),
                        "-m",
                        matchesDirectory,
                        "-o",
                        reconstructionDirectory,
                    ]
                    + globalSFMOptions,
                }
            )
        if pipelineType == "incremental":
            commands.append(
                {
                    "title": "Do incremental/sequential reconstruction",
                    "command": [
                        os.path.join(openmvgBin, "openMVG_main_SfM"),
                        "-s",
                        "INCREMENTAL",
                        "-i",
                        os.path.join(matchesDirectory, "sfm_data.json"),
                        "-m",
                        matchesDirectory,
                        "-o",
                        reconstructionDirectory,
                    ]
                    + incrementalSFMOptions,
                }
            )

        if pipelineType == "incremental2":
            commands.append(
                {
                    "title": "Do incremental/sequential reconstruction",
                    "command": [
                        os.path.join(openmvgBin, "openMVG_main_SfM"),
                        "-s",
                        "INCREMENTALV2",
                        "-i",
                        os.path.join(matchesDirectory, "sfm_data.json"),
                        "-m",
                        matchesDirectory,
                        "-o",
                        reconstructionDirectory,
                    ]
                    + incrementalSFMOptions,
                }
            )

        if args.colorize:
            commands.append(
                {
                    "title": "Colorize sparse point cloud",
                    "command": [
                        os.path.join(openmvgBin, "openMVG_main_ComputeSfM_DataColor"),
                        "-i",
                        os.path.join(reconstructionDirectory, "sfm_data.bin"),
                        "-o",
                        os.path.join(reconstructionDirectory, "colorized.ply"),
                    ],
                }
            )

    if args.run_colmap:
        # colmap feature_extractor \
        # --SiftExtraction.use_gpu $use_gpu \
        # --ImageReader.camera_model OPENCV \
        # --database_path $database_folder \
        # --image_path $images_folder \
        # TODO we delete the use_gpu camera option here
        commands.append(
            {
                "title": "Colmap feature_extractor",
                "command": [
                    os.path.join(colmapBin, "colmap"),
                    "feature_extractor",
                    "--database_path",
                    colmap_database_folder,
                    "--image_path",
                    colmap_images_folder,
                ]
                + colmap_feature_extractor_options,
            }
        )
        # colmap exhaustive_matcher \
        # --SiftMatching.use_gpu $use_gpu \
        # --database_path $database_folder \
        # TODO we delete the use_gpu camera option here
        commands.append(
            {
                "title": "colmap exhaustive_matcher",
                "command": [
                    os.path.join(colmapBin, "colmap"),
                    "exhaustive_matcher",
                    "--database_path",
                    colmap_database_folder,
                ]
                + colmap_exhaustive_matcher_options,
            }
        )
        # colmap mapper \
        # --database_path $database_folder \
        # --image_path $images_folder \
        # --output_path $output_folder \
        commands.append(
            {
                "title": "colmap mapper",
                "command": [
                    os.path.join(colmapBin, "colmap"),
                    "mapper",
                    "--database_path",
                    colmap_database_folder,
                    "--image_path",
                    colmap_images_folder,
                    "--output_path",
                    colmap_output_folder,
                ]
                + colmap_mapper_options,
            }
        )
        # colmap image_undistorter \
        # --image_path $images_folder \
        # --input_path $output_folder/0 \
        # --output_path $working_folder/dense \
        # --output_type COLMAP \
        # TODO the output type need redefine
        commands.append(
            {
                "title": "colmap image_undistorter",
                "command": [
                    os.path.join(colmapBin, "colmap"),
                    "image_undistorter",
                    "--image_path",
                    colmap_images_folder,
                    "--input_path",
                    os.path.join(colmap_output_folder, "0"),
                    "--output_path",
                    os.path.join(colmap_working_folder, "dense"),
                    "--output_type",
                    "COLMAP",
                ]
                + colmap_image_undistorter_options,
            }
        )
        # colmap model_converter \
        # --input_path $working_folder/dense/sparse \
        # --output_path $working_folder/dense/sparse \
        # --output_type TXT
        # TODO the output type here need redifne
        commands.append(
            {
                "title": "colmap model_converter",
                "command": [
                    os.path.join(colmapBin, "colmap"),
                    "model_converter",
                    "--input_path",
                    os.path.join(colmap_working_folder, "dense", "sparse"),
                    "--output_path",
                    os.path.join(colmap_working_folder, "dense", "sparse"),
                    "--output_type",
                    "TXT",
                ]
                + colmap_model_converter_options,
            }
        )

    if args.run_openmvs:
        sceneFileName = ["model_scene"]

        # sudo ./InterfaceCOLMAP \
        # --working-folder $working_folder \
        # -i $working_folder/dense/ \
        # --output-file $working_folder/model_colmap.mvs

        commands.append(
            {
                "title": "Convert Colmap project to OpenMVS",
                "command": [
                    os.path.join(openmvsBin, "InterfaceCOLMAP"),
                    "-i",
                    os.path.join(colmap_working_folder, "dense"),
                    "--output-file",
                    os.path.join(MVSDirectory, "scene.mvs"),
                ],
            }
        )

        commands.append(
            {
                "title": "Densify point cloud",
                "command": [
                    os.path.join(openmvsBin, "DensifyPointCloud"),
                    "--input-file",
                    os.path.join(MVSDirectory, "scene.mvs"),
                    "--working-folder",
                    os.path.join(colmap_working_folder),
                    "--output-file",
                    os.path.join(MVSDirectory, "densify.mvs"),
                ]
                + densifyPointCloudOptions,
            }
        )

        if True:
            mvsFileName = "_".join(sceneFileName) + ".mvs"
            commands.append(
                {
                    "title": "Reconstruct mesh",
                    "command": [
                        os.path.join(openmvsBin, "ReconstructMesh"),
                        "--input-file",
                        os.path.join(MVSDirectory, "densify.mvs"),
                        "--working-folder",
                        os.path.join(colmap_working_folder),
                        "--output-file",
                        os.path.join(MVSDirectory, "reconstruct.mvs"),
                    ]
                    + reconstructMeshOptions,
                }
            )

            rmCudaOk = False
            if args.rmcuda:
                if os.path.exists(os.path.join(openmvsBin, "RefineMeshCUDA")):
                    rmCudaOk = True
            if rmCudaOk:
                commands.append(
                    {
                        "title": "Refine mesh using CUDA",
                        "command": [
                            os.path.join(openmvsBin, "RefineMeshCUDA"),
                            "--input-file",
                            os.path.join(MVSDirectory, "reconstruct.mvs"),
                            "--working-folder",
                            os.path.join(colmap_working_folder),
                            "--output-file",
                            os.path.join(MVSDirectory, "refine.mvs"),
                        ]
                        + refineMeshOptions,
                    }
                )
            else:
                commands.append(
                    {
                        "title": "Refine mesh",
                        "command": [
                            os.path.join(openmvsBin, "RefineMesh"),
                            "--input-file",
                            os.path.join(MVSDirectory, "reconstruct.mvs"),
                            "--working-folder",
                            os.path.join(colmap_working_folder),
                            "--output-file",
                            os.path.join(MVSDirectory, "refine.mvs"),
                        ]
                        + refineMeshOptions,
                    }
                )
            sceneFileName.append("refine")

            mvsFileName = "_".join(sceneFileName) + ".mvs"
            commands.append(
                {
                    "title": "Texture mesh",
                    "command": [
                        os.path.join(openmvsBin, "TextureMesh"),
                        "--input-file",
                        os.path.join(MVSDirectory, "refine.mvs"),
                        "--working-folder",
                        os.path.join(colmap_working_folder),
                        "--output-file",
                        os.path.join(MVSDirectory, "result.obj"),
                        "--export-type",
                        "obj",
                    ]
                    + textureMeshOptions,
                }
            )

    if args.debug:
        for instruction in commands:
            print(instruction["title"])
            print(
                "========================================================================="
            )
            print(" ".join(map(str, instruction["command"])))
            print("")
        sys.exit()
    else:
        if args.run_openmvg and not os.path.exists(matchesDirectory):
            os.makedirs(matchesDirectory)
        if args.run_openmvg and not os.path.exists(reconstructionDirectory):
            os.makedirs(reconstructionDirectory)
        if args.run_openmvs and not os.path.exists(MVSDirectory):
            os.makedirs(MVSDirectory)
        if args.run_colmap and not os.path.exists(colmap_working_folder):
            os.makedirs(colmap_working_folder)
        if args.run_colmap and not os.path.exists(colmap_images_folder):
            os.makedirs(colmap_images_folder)
        if args.run_colmap and not os.path.exists(colmap_output_folder):
            os.makedirs(colmap_output_folder)

    return commands


def runCommandTest(cmd):
    cwd = outputDirectory
    if "OpenMVS" in cmd[0]:
        cwd = MVSDirectory
    try:
        p = subprocess.Popen(cmd, cwd=cwd)
        p.communicate()
        return p.returncode
    except OSError as err:
        if err.errno == errno.ENOENT:
            logging.exception("An exception was thrown in runCommand(cmd)")
            print(
                "Could not find executable: {0}  \nHave you installed all the requirements?".format(
                    cmd[0]
                )
            )
        else:
            print("Could not run command: {0}".format(err))
        return -1
    except:
        print("Could not run command")
        return -1


def runCommand(cmd):
    cwd = outputDirectory
    if "OpenMVS" in cmd[0]:
        cwd = MVSDirectory
    try:
        p = subprocess.Popen(cmd, cwd=cwd)
        p.communicate()
        return p.returncode
    except OSError as err:
        if err.errno == errno.ENOENT:
            print(
                "Could not find executable: {0} - Have you installed all the requirements?".format(
                    cmd
                )
            )
        else:
            print("Could not run command 2: {0}".format(err))
        return -1
    except Exception as e:
        logging.exception("An exception was thrown in runCommand(cmd)")
        return -1


def runCommands(commands):
    startTime = int(time.time())
    for instruction in commands:
        print(instruction["title"])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Running Command:")
        print(" ".join(map(str, instruction["command"])))
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        rc = runCommandTest(map(str, instruction["command"]))
        if rc != 0:
            print("Failed while executing: ")
            print(" ".join(map(str, instruction["command"])))
            sys.exit(1)
    endTime = int(time.time())

    timeDifference = endTime - startTime
    hours = int(math.floor(timeDifference / 60 / 60))
    minutes = int(math.floor((timeDifference - hours * 60 * 60) / 60))
    seconds = int(math.floor(timeDifference - (hours * 60 * 60 + minutes * 60)))
    print(
        "\n\nFinished without errors (I guess) - Time used:: {0}:{1}:{2}".format(
            ("00" + str(hours))[-2:],
            ("00" + str(minutes))[-2:],
            ("00" + str(seconds))[-2:],
        )
    )


parser = createParser()
args = parser.parse_args()
commands = createCommands(args)
runCommands(commands)
