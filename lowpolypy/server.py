import logging

import cv2
import numpy as np
from asgiref.sync import sync_to_async
from fastapi import FastAPI, File, Response, UploadFile

from .run import run

logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/")
async def read_root():
    return "LowPolyPy API running!"


@app.post(
    "/lowpoly",
    responses={
        # Manually specify a possible response with our custom media type.
        200: {"content": {"application/octet-stream": {}}}
    },
)
async def process_image(
    file: UploadFile = File(...),
    conv_points_num_points: int = 1000,
    conv_points_num_filler_points: int = 300,
    weight_filler_points: bool = True,
    output_size: int = 2560,
):
    logger.debug(f"Received image: {file.filename=} {file.content_type=} {file.size=}")
    # Load the image bytes as a numpy byte array
    byte_array = np.frombuffer(await file.read(), np.uint8)
    # Decode the image bytes to an image
    logger.debug("Decoding image...")
    image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)

    logger.debug("Processing image...")
    lowpoly_image_array = await sync_to_async(run)(
        image=image,
        conv_points_num_points=conv_points_num_points,
        conv_points_num_filler_points=conv_points_num_filler_points,
        weight_filler_points=weight_filler_points,
        output_size=output_size,
    )

    # Write the image to a byte buffer
    logger.debug("Encoding image...")
    _, buffer = cv2.imencode(
        ".png", lowpoly_image_array, [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )

    return Response(buffer.tobytes(), media_type="image/png")
