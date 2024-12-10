#pragma once
#include <lux/engine/ui/Widget.hpp>
#include <lux/engine/ui/Window.hpp>
#include <lux/engine/ui/RenderContext.hpp>
#include <lux/engine/gapi/vk/vk.hpp>
#include <lux/engine/gapi/vk/ShaderLoader.hpp>
#include <lux/engine/math/eigen_extend.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <stdexcept>
#include <array>
#include <cassert>

#ifndef __PI_FLOAT_CONSTANT
#   define __PI_FLOAT_CONSTANT 3.14159265358979323846F
#endif

struct PointVertex {
	Eigen::Vector3f position; // x, y, z
	Eigen::Vector3f color;    // r, g, b

	static const VkVertexInputBindingDescription& getBindingDescription() {
		static const VkVertexInputBindingDescription bindingDescription{
			.binding = 0,
			.stride = sizeof(PointVertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		};
		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 2>& getAttributeDescriptions() {
		static std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{
			VkVertexInputAttributeDescription{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(PointVertex, position)
			},
			VkVertexInputAttributeDescription{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(PointVertex, color)
			}
		};
		return attributeDescriptions;
	}
};

class PointCloudScene : public lux::ui::Widget {
public:
	PointCloudScene(lux::ui::Window& window);
	~PointCloudScene();

private:
	void createCommandPool();
	void createRenderPass();
	void createDepthResources();
	void createUniformBuffer();
	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipelineLayout(const std::vector<VkDescriptorSetLayout>&);
	void loadShaders();
	void createPipeline();
	void createViewportResources();
	void recreateViewportResources();
	void destroyViewportResources();
	void bindMVPMatrixToShader();
	void issueDrawCommands();
	void renderPointCloud();
	void paint() override;

	void createVertexBuffer(const std::vector<PointVertex>&);
	void destroyVertexBuffer();

	std::vector<VkImage>			viewport_images_;
	std::vector<VkDeviceMemory>		dst_image_memory_;
	std::vector<VkImageView>		viewport_image_views_;

	VkImage                         depth_image_{ VK_NULL_HANDLE };
	VkDeviceMemory                  depth_image_memory_{ VK_NULL_HANDLE };
	VkImageView                     depth_image_view_{ VK_NULL_HANDLE };

	lux::gapi::vk::CommandPool		command_pool_;
	lux::gapi::vk::RenderPass		render_pass_;

	lux::gapi::vk::Buffer           uniform_buffer_;
	VkDeviceMemory					uniform_buffer_memory_{ VK_NULL_HANDLE };
	void*							uniformBuffersMapped{ nullptr };

	lux::gapi::vk::Buffer           vertex_buffer_;
	VkDeviceMemory					vertex_buffer_memory_{ VK_NULL_HANDLE };
	void*							vertexBufferMapped{ nullptr };

	lux::gapi::vk::FrameBuffer		frame_buffer_;
	lux::gapi::vk::ShaderModule		vertex_shader_;
	lux::gapi::vk::ShaderModule		fragment_shader_;
	lux::gapi::vk::PipelineLayout   pipeline_layout_;
	lux::gapi::vk::GraphicsPipeline pipeline_;

	VkDescriptorSet                 tex_id_;

	VkDescriptorSetLayout           descriptor_set_layout_{ VK_NULL_HANDLE };
	VkDescriptorSet                 descriptor_set_{ VK_NULL_HANDLE };
	VkSampler                       sampler_{ VK_NULL_HANDLE };

	int width_{ 1 };
	int height_{ 1 };
	int prev_width_{ 1 };
	int prev_height_{ 1 };
};
