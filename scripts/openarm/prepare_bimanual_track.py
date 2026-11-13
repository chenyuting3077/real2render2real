"""Prepare video-derived hand timelines and antipodal grasps, without starting Isaac Sim."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--dig-config", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True, help="centred metric grasp mesh for the tracked object")
    parser.add_argument("--output", type=Path, required=True, help="new metadata JSON; original track/grasp files are untouched")
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--video", type=Path, default=None, help="tracked video; otherwise use track/cache_info.json")
    parser.add_argument("--source-fps", type=float, default=None, help="explicit FPS if video metadata is unavailable")
    parser.add_argument("--contact-enter", type=float, default=0.04)
    parser.add_argument("--contact-leave", type=float, default=0.055)
    parser.add_argument("--max-grasp-width", type=float, default=0.088)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve prior metadata")
    from real2render2real.isaaclab_viser.openarm_simulators.video_bimanual import prepare_bimanual_data

    transforms = args.dig_config.parent / "dataparser_transforms.json"
    scale = json.loads(transforms.read_text())["scale"]
    video = args.video
    cache = args.track.parent / "cache_info.json"
    if video is None and cache.exists():
        video = Path(json.loads(cache.read_text())["video_path"])
    fps = args.source_fps
    if fps is None and video is not None and video.is_file():
        import cv2

        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
    if fps is None or fps <= 0:
        parser.error("Video FPS is unknown; provide --video or --source-fps")
    data = prepare_bimanual_data(args.track, scale, args.mesh, object_index=args.object_index,
                                 enter=args.contact_enter, leave=args.contact_leave, max_width=args.max_grasp_width,
                                 source_fps=float(fps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for side, hand in data["hands"].items():
        print(f"{side}: inferred source-frame contact intervals {hand['intervals']}; "
              f"{len(hand['grasp_candidates'])} video-ranked antipodal candidates")
    print(f"Saved {args.output}. Object trajectory is unchanged; robot feasibility is not yet validated.")


if __name__ == "__main__":
    main()
