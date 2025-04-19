from influxdb import UploadToInflux

def main():
    try:
        print("Starting data update...")
        UploadToInflux.upload_after_last_upload()
        print("Data update completed.")
    except Exception as e:
        print(f"Error during data update: {str(e)}")

if __name__ == "__main__":
    main()