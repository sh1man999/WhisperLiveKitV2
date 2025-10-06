#!/usr/bin/env python3
import time
import logging
from typing import Union

from .backends import FasterWhisperASR
from whisperlivekit.warmup import warmup_asr

logger = logging.getLogger(__name__)



WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(
    ","
)


def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert (
        lan in WHISPER_LANG_CODES
    ), "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    # supported by fast-mosestokenizer
    if (
        lan
        in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split()
    ):
        from mosestokenizer import MosesSentenceSplitter        

        return MosesSentenceSplitter(lan)

    # the following languages are in Whisper, but not in wtpsplit:
    if (
        lan
        in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split()
    ):
        logger.debug(
            f"{lan} code is not supported by wtpsplit. Going to use None lang_code option."
        )
        lan = None

    from wtpsplit import WtP

    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")

    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)

    return WtPtok()


def backend_factory(
            lan,
            model_size,
            model_cache_dir,
            model_dir,
            buffer_trimming,
            buffer_trimming_sec,
            confidence_validation,
            warmup_file=None,
            min_chunk_size=None,
            device="auto",
            device_index: Union[int, list[int]] = 0,
            cpu_threads: int = 0,
            num_workers: int = 1
        ):

    t = time.time()
    logger.info(f"Loading Whisper {model_size} model for language {lan}...")
    asr = FasterWhisperASR(
        model_size=model_size,
        lan=lan,
        cache_dir=model_cache_dir,
        model_dir=model_dir,
        device=device,
        device_index=device_index,
        cpu_threads=cpu_threads,
        num_workers=num_workers
    )
    e = time.time()
    logger.info(f"done. It took {round(e - t, 2)} seconds.")
    #asr.use_vad()

    # Create the tokenizer
    if buffer_trimming == "sentence":
        tokenizer = create_tokenizer(lan)
    else:
        tokenizer = None
    
    warmup_asr(asr, warmup_file)
    
    asr.confidence_validation = confidence_validation
    asr.tokenizer = tokenizer
    asr.buffer_trimming = buffer_trimming
    asr.buffer_trimming_sec = buffer_trimming_sec
    return asr