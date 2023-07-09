import xmltodict
from argparse import ArgumentParser
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from datetime import datetime
import os
from xml.dom.minidom import parseString

def clip_mp4(src: str,
             start: str,
             end: str,
             trg_dir: str) -> dict:
    """Segment the original mp4 into multiple mp4 files.

    Params:
    src: Source mp4 file
    start: Start time of the segment
    end: End time of the segment
    trg_dir: Target directory, e.g., "1".

    Returns:
    dict: A dict of the information of the segmented audio for
    creating the output .eaf file.
    """
    # divide by 1000 to adjust the input format to moviepy
    if not os.path.exists(trg_dir):
        os.mkdir(trg_dir)
    trg = trg_dir + "/" + src[:-4].split("/")[-1] + "_{}_{}.mp4".format(start, end)
    ffmpeg_extract_subclip(src,
                           int(start) / 1000,
                           int(end) / 1000,
                           targetname=trg)
    trg_abspath = "file://" + os.path.abspath(trg)
    trg_relpath = os.path.relpath(trg)
    output = {"src": src,
              "start": start,
              "end": end,
              "abspath": trg_abspath,
              "relpath": trg_relpath}
    return output

def eaf2segments(eaf_file: str) -> tuple:
    """
    Segment the original eaf file into multiple eaf files per annotation.

    Params:
    eaf_file: Path to the eaf original file.

    Returns:
    tuple: A tuple of a list of annotations and a list of timestamps.
    """
    with open(eaf_file) as f:
        xml = f.read()
    dct = xmltodict.parse(xml)
    annotations = dct["ANNOTATION_DOCUMENT"]["TIER"]["ANNOTATION"]
    timestamps = dct["ANNOTATION_DOCUMENT"]["TIME_ORDER"]["TIME_SLOT"]
    return dct, annotations, timestamps

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", type=str,
                        help="Input the working directory name, e.g., Chapter1")
    parser.add_argument("-e", "--eaf", type=str,
                        help="Input original .eaf file.")
    parser.add_argument("-a", "--audio", type=str,
                        help="Input original .mp4 file.")
    parser.add_argument("-D", "--debug", action='store_true',
                        help="Debug mode if specified")
    args = parser.parse_args()
    assert os.path.exists(args.directory + "/" + args.eaf)
    assert os.path.exists(args.directory + "/" + args.audio)

    dct, annotations, timestamps = eaf2segments(args.directory + "/" + args.eaf)

    for i in range(len(annotations)):
        if args.debug:
            if i == 1:
                break
        stamp = i * 2
        start = timestamps[stamp]["@TIME_VALUE"]
        end = timestamps[stamp+1]["@TIME_VALUE"]

        text = annotations[i]["ALIGNABLE_ANNOTATION"]["ANNOTATION_VALUE"]
        annotation_id = dct["ANNOTATION_DOCUMENT"]["HEADER"]["PROPERTY"][1]["#text"]

        output = clip_mp4(args.directory + "/" + args.audio,
                          start,
                          end,
                          trg_dir=args.directory + "/" + str(i+1))
        now = datetime.now()
        now = now.astimezone().isoformat("T", "seconds")

        eaf_text = """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="" DATE="2023-06-05T23:42:28-05:00"
    FORMAT="3.0" VERSION="3.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <MEDIA_DESCRIPTOR
            MEDIA_URL="{}"
            MIME_TYPE="video/mp4" RELATIVE_MEDIA_URL="{}"/>
        <PROPERTY NAME="lastUsedAnnotationId">{}</PROPERTY>
    </HEADER>
    <TIME_ORDER>
        <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
        <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="{}"/>
    </TIME_ORDER>
    <TIER LINGUISTIC_TYPE_REF="default-lt" TIER_ID="default">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1"
                TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
                <ANNOTATION_VALUE>{}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false"
        LINGUISTIC_TYPE_ID="default-lt" TIME_ALIGNABLE="true"/>
    <CONSTRAINT
        DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
    <CONSTRAINT
        DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered" STEREOTYPE="Symbolic_Subdivision"/>
    <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
    <CONSTRAINT
        DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
</ANNOTATION_DOCUMENT>
""".format(output["abspath"],
           output["relpath"],
           annotation_id,
           str(int(end) - int(start)),
           text)

        output_eaf = "{}/{}/{}.eaf".format(args.directory, str(i+1), str(i+1))
        with open(output_eaf, "w", encoding="utf-8") as f:
            f.write(eaf_text)
            
