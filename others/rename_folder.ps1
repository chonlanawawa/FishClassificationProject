$folderPath = "C:\Users\chonl\Downloads\fish_dataset\raw"

# get all folder
$folders = Get-ChildItem -Path $folderPath -Directory

# rename each folder ( change spaces ' ' to '_' )
foreach ($folder in $folders) {
    $newName = $folder.Name -replace ' ', '_'  # Replace spaces with underscores
    $newPath = Join-Path -Path $folderPath -ChildPath $newName
    Rename-Item -Path $folder.FullName -NewName $newName -Force
}
