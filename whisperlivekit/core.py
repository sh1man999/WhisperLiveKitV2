from typing import Union

try:
    from whisperlivekit.whisper_streaming_custom.whisper_online import backend_factory
    from whisperlivekit.whisper_streaming_custom.online_asr import OnlineASRProcessor
except ImportError:
    from .whisper_streaming_custom.whisper_online import backend_factory
from argparse import Namespace

class TranscriptionEngine:
    
    def __init__(self,
                 is_diarization: bool=False,
                 punctuation_split: bool=False,
                 split_on_punctuation_for_display: bool=False,
                 vac: bool=True,
                 vac_chunk_size: float=0.04,
                 log_level: str= "DEBUG",
                 vad: bool=True,
                 pcm_input:bool=False,
                 disable_punctuation_split:bool=False,
                 diarization_backend: str="diart",
                 warmup_file: str=None,
                 min_chunk_size: float=0.5,
                 model_size: str="tiny",
                 model_cache_dir: str=None,
                 model_dir: str=None,
                 device="auto",
                 device_index: Union[int, list[int]] = 0,
                 cpu_threads: int = 0,
                 num_workers: int = 1,
                 lan: str="auto",
                 buffer_trimming: str= "segment",
                 confidence_validation: bool= False,
                 buffer_trimming_sec: int= 15,
                 segmentation_model_name: str= "pyannote/segmentation-3.0",
                 embedding_model_name: str = "pyannote/embedding",
                 compute_type: str = "float16"
                 ):

        self.args = Namespace(
            diarization=is_diarization,
            punctuation_split=punctuation_split,
            split_on_punctuation_for_display=split_on_punctuation_for_display,
            vac=vac,
            vac_chunk_size=vac_chunk_size,
            log_level=log_level,
            vad=vad,
            pcm_input=pcm_input,
            disable_punctuation_split=disable_punctuation_split,
            diarization_backend=diarization_backend,
            warmup_file=warmup_file,
            min_chunk_size=min_chunk_size,
            model_size=model_size,
            model_cache_dir=model_cache_dir,
            model_dir=model_dir,
            lan=lan,
            buffer_trimming=buffer_trimming,
            confidence_validation=confidence_validation,
            buffer_trimming_sec=buffer_trimming_sec,
            segmentation_model_name=segmentation_model_name,
            embedding_model_name=embedding_model_name,
            transcription=True
        )


        self.asr = None
        self.tokenizer = None
        self.diarization = None
        self.vac_model = None
        
        if vac:
            import torch
            self.vac_model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")            


        self.asr = backend_factory(
            lan=lan,
            model_size=model_size,
            model_cache_dir=model_cache_dir,
            model_dir=model_dir,
            buffer_trimming=buffer_trimming,
            buffer_trimming_sec=buffer_trimming_sec,
            confidence_validation=confidence_validation,
            warmup_file=warmup_file,
            min_chunk_size=min_chunk_size,
            device=device,
            device_index=device_index,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            compute_type=compute_type
        )

        if is_diarization:
            if diarization_backend == "diart":
                from whisperlivekit.diarization.diart_backend import DiartDiarization
                self.diarization_model = DiartDiarization(
                    block_duration=min_chunk_size,
                    segmentation_model_name=segmentation_model_name,
                    embedding_model_name=embedding_model_name,
                )
            elif diarization_backend == "sortformer":
                from whisperlivekit.diarization.sortformer_backend import SortformerDiarization
                self.diarization_model = SortformerDiarization()

  
def online_diarization_factory(args, diarization_backend):
    if args.diarization_backend == "diart":
        online = diarization_backend
        # Not the best here, since several user/instances will share the same backend, but diart is not SOTA anymore and sortformer is recommended
    
    if args.diarization_backend == "sortformer":
        from whisperlivekit.diarization.sortformer_backend import SortformerDiarizationOnline
        online = SortformerDiarizationOnline(shared_model=diarization_backend)
    return online