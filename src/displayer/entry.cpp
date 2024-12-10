#include <lux/engine/ui/Window.hpp>
#include "displayer/PointCloudScene.hpp"
#include <iostream>

int main(int argc, char** argv) {
	using namespace lux::ui;

	auto window = std::make_unique<Window>(1920, 1080, "pcd");
	auto pcd    = std::make_unique<PointCloudScene>(*window);

	if (!window->init()) {
		std::cerr << "Failed to initialize window" << std::endl;
		return -1;
	}

	return window->exec();
}
