import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tyro

from nerfstudio.process_data.video_to_nerfstudio_dataset import VideoToNerfstudioDataset
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.external_methods import ExternalMethodDummyTrainerConfig, get_external_methods
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import PhototourismDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.datasets.sdf_dataset import SDFDataset
from nerfstudio.data.datasets.semantic_dataset import SemanticDataset
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.models.generfacto import GenerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.models.neus_facto import NeuSFactoModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.registry import discover_methods
from nerfstudio.scripts.exporter import ExportPoissonMesh

from sa3d.sa3d_datamanager import SA3DDataManagerConfig
from sa3d.sa3d import SA3DModelConfig
from sa3d.sa3d_pipeline import SA3DPipelineConfig
from sa3d.sa3d_trainer import SA3DTrainerConfig
from sa3d.sa3d_optimizer import SGDOptimizerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from sa3d.self_prompting.sam3d import SAM3DConfig
from sa3d.sa3d_field import TCNNMaskFieldConfig

import nerfstudio.scripts.train as trainer
from nerfstudio.utils.rich_utils import CONSOLE
from abc import ABC, abstractmethod




def video2mesh(
         input_folder : Path ,
         output_folder: Path,
):
        preoutput_folder = Path("~/preprocessed_data/").expanduser()
        os.makedirs(preoutput_folder, exist_ok=True)
        CONSOLE.print("loading videos")
        init_output_folder = output_folder
        for filename in os.listdir(input_folder):
            if filename.endswith(('.MP4', '.AVI', '.MOV', '.MKV')):
                input_path = Path(os.path.join(input_folder, filename))
                output_path = Path(os.path.join(preoutput_folder, os.path.splitext(filename)[0]+"/"))

                print(f"Processing {input_path} and saving to {output_path}")
                
                colmap_generator = VideoToNerfstudioDataset(num_frames_target=150, data=input_path, output_dir = output_path)
                colmap_generator.main()
                nerfConfig = TrainerConfig(
                    method_name="nerfacto",
                    steps_per_eval_batch=500,
                    steps_per_save=2000,
                    max_num_iterations=30000,
                    mixed_precision=True,
                    pipeline=VanillaPipelineConfig(
                        datamanager=ParallelDataManagerConfig(
                            dataparser=NerfstudioDataParserConfig(),
                            train_num_rays_per_batch=4096,
                            eval_num_rays_per_batch=4096,
                        ),
                        model=NerfactoModelConfig(
                            eval_num_rays_per_chunk=1 << 15,
                            average_init_density=0.01,
                            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                        ),
                    ),
                    optimizers={
                        "proposal_networks": {
                            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
                        },
                        "fields": {
                            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
                        },
                        "camera_opt": {
                            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
                        },
                    },
                    viewer=ViewerConfig(num_rays_per_chunk=1 << 15,quit_on_train_completion=True),
                    vis="viewer",
                    data=output_path,
                    output_dir=Path(os.path.join("~/outputs/")).expanduser()
                )
                
                nerf_dir = trainer.main(nerfConfig)
                nerf_dir = nerf_dir / "nerfstudio_models"

                sa3d_config = config=SA3DTrainerConfig(
                    method_name="sa3d",
                    max_num_iterations=150,
                    save_only_latest_checkpoint=True,
                    mixed_precision=False,
                    pipeline=SA3DPipelineConfig(
                        text_prompt='',
                        datamanager=SA3DDataManagerConfig(
                            dataparser=NerfstudioDataParserConfig(),
                            train_num_rays_per_batch=2048,
                            eval_num_rays_per_batch=2048,
                        ),
                        model=SA3DModelConfig(
                            mask_fields=TCNNMaskFieldConfig(
                                base_res=128,
                                num_levels=16,
                                max_res=2048,
                                log2_hashmap_size=19,
                                mask_threshold=1e-1
                            ),
                            eval_num_rays_per_chunk=1 << 11,
                            remove_mask_floaters=True,
                            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                            average_init_density=0.01,
                            predict_normals=True,
                        ),
                        network=SAM3DConfig(
                            num_prompts=10,
                            neg_lamda=1.0
                        )
                    ),
                    optimizers={
                        "mask_fields": {
                            "optimizer": SGDOptimizerConfig(lr=1e-1),
                            "scheduler": None,
                        },
                    },
                    viewer=ViewerConfig(num_rays_per_chunk=1 << 15,quit_on_train_completion=True),
                    vis="viewer",
                    data=output_path,
                    output_dir=Path("~/sa3d").expanduser() / os.path.splitext(filename)[0],
                    load_dir=nerf_dir
                )
                sa3d_dir = trainer.main(sa3d_config)
                sa3d_dir = sa3d_dir / "config.yml"
                output_folder = output_folder.expanduser() / os.path.splitext(filename)[0]
                os.makedirs(output_folder, exist_ok=True)
                mesh_generator = ExportPoissonMesh(load_config=sa3d_dir,output_dir=output_folder, std_ratio=2.0)
                mesh_generator.main()
                output_folder = init_output_folder


def entrypoint():
    tyro.cli(
       video2mesh
    )

if __name__ == "__main__":
    entrypoint()

