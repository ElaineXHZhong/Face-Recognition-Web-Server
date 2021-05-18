# API Specification

### Main Server

| API No.  |          API         |   Method  |                      Functionality                        |
|:--------:|:--------------------:|:---------:|-----------------------------------------------------------|
|    R1    |         `/`          |    GET    | select prediction mode: video predict, single image predict, batch image predict, find similar identity |
|    R2    |      `/trainModel`   |    GET    | page for uploading the compressed training set of still image or video |
|    R3    |     `/trainExplain`  |    GET    | instruction page to instruct users how to organize training set in zip format |
|    R4    |     `/trainByImage`  |    GET    | manually upload compressed training set of still image and direct to process server API `/` |
|    R5    |     `/trainByVideo`  |    GET    | manually upload compressed training set of video and direct to process server API `/` |
|    R6    | `/uploadVideoPage`   |    GET    | manually upload KOL video for real-time face recognition |
|    R7    |`/predictVideoResult` | GET, POST | display page of real-time face recognition of face frame from video |
|    R8    | `/predictSinglePage` |    GET    | manually upload single image file for identity prediction |
|    R9    | `/predictSingleImage`| GET, POST | get single image prediction result |
|    R10   | `/predictBatchPage`  |    GET    | manually upload multiple image files for identity prediction |
|    R11   | `/predictBatchImage` | GET, POST | get multiple image prediction results |
|    R12   | `/findSimilarKOLPage`|    GET    | manually upload single image file to find tip k similar identities |
|    R13   | `/findSimilarKOLResult` | GET, POST | get tip k similar identities |

### Process Server

| API No.  |          API         |   Method  |                      Functionality                        |
|:--------:|:--------------------:|:---------:|-----------------------------------------------------------|
|    R1    | `/`                  |    GET    | navigation page for user to complete the processing phase of training |
|    R2    | `/processingProgress`| GET, POST | display page which shows user training progress |
|    R3    | `/align`             | GET, POST | guide user to complete the 'align faces' action and see alignment result |
|    R4    | `/clean`             | GET, POST | guide user to complete the 'clean faces' action and see cleaning result |
|    R5    | `/import_clean`      | GET, POST | guide user to select which kol to import to the Training Pool           |
|    R6    | `/import_result`     | GET, POST | import selected kol to Training Pool and display pool summary |
|    R7    | `/train_model`       | GET, POST | guide user to complete the 'train model' action and see training result |

### Video Server

| API No.  |          API         |   Method  |                      Functionality                        |
|:--------:|:--------------------:|:---------:|-----------------------------------------------------------|
|    R1    | `/`                  |    GET    | manually upload video file for identity prediction        |
|    R2    | `/video_feed`        | GET, POST | get real-time face identity prediction result of uploaded video |