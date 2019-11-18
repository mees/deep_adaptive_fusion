image_db = '/home/meeso/day_night_people';
image_filenames = textread([image_db '/data/ImageSets/val.txt'], '%s', 'delimiter', '\n');
for i = 1:length(image_filenames)
    if exist([image_db '/data/ImagesQhd/' image_filenames{i} '.jpg'], 'file') == 2
	image_filenames{i} = [image_db '/data/ImagesQhd/' image_filenames{i} '.jpg'];
    end
    if exist([image_db '/data/ImagesQhd/' image_filenames{i} '.png'], 'file') == 2
        image_filenames{i} = [image_db '/data/ImagesQhd/' image_filenames{i} '.png'];
    end
end
selective_search_rcnn(image_filenames, '/home/meeso/day_night_people/selective_search/val.mat');
