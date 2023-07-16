# Whole Slide Image Patch Extraction

This project focuses on extracting patches from Whole Slide Images (WSI) using various extraction methods. It provides a
modular and configurable approach to extract patches based on different criteria such as point annotations and tiles.

## Project Structure

The project consists of the following components:

- `SVSLoader` package: Contains the code for loading and managing Whole Slide Images.

- `Processing` package: Contains the code for different patch extraction methods and related utilities.

- `Config` package: Contains configuration-related files and utilities.

- `Utils` package: Contains utility functions used across the project.

- `main.py`: The main script to run the patch extraction process.

## Setup and Dependencies

To run the project, ensure that you have the following dependencies installed:

- Python (version 3.8)
- OpenCV (version X.X.X)
- Numpy (version X.X.X)
- PIL (version X.X.X)
- BeautifulSoup4 (version X.X.X)

## Usage

1. Set up the project by cloning the repository and installing the required dependencies.

2. Prepare the Whole Slide Images (WSI) and associated files:
    - Place the WSI files in the designated directory.
    - Ensure the associated annotation files match the specified pattern.

3. Configure the extraction settings:
    - Update the configuration file (`config.yaml`) in the `Config` directory according to your needs.
    - Adjust the extraction module, patches directory, patch size, scaling factor, and other parameters as required.

4. Run the patch extraction process:
    - Open a terminal and navigate to the project directory.
    - Execute the main script using the following command:

      ```
      python main.py -c config/config.yaml
      ```

      Make sure to provide the correct path to the configuration file.

5. Monitor the progress and results:
    - The patch extraction process will start, and the progress will be displayed in the console.
    - Extracted patches will be saved in the specified directory.

## Configuration

The `config.yaml` file in the `Config` directory allows you to customize the patch extraction process. It includes the
following sections:

### Whole Slide Image loader configuration

- `WSL_DATA_DIR`: The directory path where the WSI files are located.
- `ASSOCIATED_FILE_PATTERN`: The regular expression pattern used to load the associated annotation files.

### Patch Extraction Configuration

- `EXTRACTION_MODULE`: The name of the patch extraction module to be used.
- `PATCHES_DIR`: The directory path where the extracted patches will be saved.
- `PATCH_SIZE`: The size of the patches to be extracted, specified as [width, height].
- `SCALING_FACTOR`: The scaling factor applied to the patch size for improved resolution.
- `USE_CIR_MASK`: A boolean value indicating whether to use a circular mask for the patches.
- `CONTEXT_MASK_RADIUS`: The radius of the circular mask applied to the patches.

## Contributing

Contributions to this project are welcome. You can contribute by:

- Reporting issues or bugs.
- Suggesting new features or enhancements.
- Submitting pull requests to fix issues, add features, or improve the code.

Please refer to the [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the [MIT License](LICENSE).

