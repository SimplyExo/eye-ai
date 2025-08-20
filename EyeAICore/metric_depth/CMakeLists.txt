project(metric_depth)

add_executable(PrepareDataset
	"prepare_dataset/main.cpp"
	"prepare_dataset/utils.cpp"
	"prepare_dataset/stb_image.cpp"
	"prepare_dataset/datasets/diode_dataset.cpp"
	"prepare_dataset/datasets/sun_rgbd_dataset.cpp"
)
target_compile_features(PrepareDataset PUBLIC cxx_std_20)
target_link_libraries(PrepareDataset PRIVATE EyeAICore)