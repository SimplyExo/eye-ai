#include "EyeAICore/utils/DepthColormap.hpp"

#include "EyeAICore/utils/Errors.hpp"
#include "EyeAICore/utils/ImageUtils.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include <algorithm>

static int inferno_depth_colormap(float relative_depth);

tl::expected<void, std::string> depth_colormap(
	std::span<const float> depth_values,
	std::span<int> colormapped_pixels
) {
	PROFILE_DEPTH_FUNCTION()

	if (depth_values.size() != colormapped_pixels.size()) {
		return tl::unexpected_fmt(
			"depth_values ({}) does not match colormapped_pixels ({})",
			depth_values.size(), colormapped_pixels.size()
		);
	}

	for (size_t i = 0; i < depth_values.size(); i++) {
		colormapped_pixels[i] = inferno_depth_colormap(depth_values[i]);
	}

	return {};
}

constexpr size_t INFERNO_COLOR_COUNT = 256;

/**
 * Inferno Colormap: index is depth (0..255)
 * value based on:
 * https://github.com/kennethmoreland-com/kennethmoreland-com.github.io/blob/master/color-advice/inferno/inferno-table-byte-0256.csv
 */
constexpr std::array<int, INFERNO_COLOR_COUNT> INFERNO_COLORS = {
	color_rgb(0, 0, 4),		  color_rgb(1, 0, 5),
	color_rgb(1, 1, 6),		  color_rgb(1, 1, 8),
	color_rgb(2, 1, 10),	  color_rgb(2, 2, 12),
	color_rgb(2, 2, 14),	  color_rgb(3, 2, 16),
	color_rgb(4, 3, 18),	  color_rgb(4, 3, 20),
	color_rgb(5, 4, 23),	  color_rgb(6, 4, 25),
	color_rgb(7, 5, 27),	  color_rgb(8, 5, 29),
	color_rgb(9, 6, 31),	  color_rgb(10, 7, 34),
	color_rgb(11, 7, 36),	  color_rgb(12, 8, 38),
	color_rgb(13, 8, 41),	  color_rgb(14, 9, 43),
	color_rgb(16, 9, 45),	  color_rgb(17, 10, 48),
	color_rgb(18, 10, 50),	  color_rgb(20, 11, 52),
	color_rgb(21, 11, 55),	  color_rgb(22, 11, 57),
	color_rgb(24, 12, 60),	  color_rgb(25, 12, 62),
	color_rgb(27, 12, 65),	  color_rgb(28, 12, 67),
	color_rgb(30, 12, 69),	  color_rgb(31, 12, 72),
	color_rgb(33, 12, 74),	  color_rgb(35, 12, 76),
	color_rgb(36, 12, 79),	  color_rgb(38, 12, 81),
	color_rgb(40, 11, 83),	  color_rgb(41, 11, 85),
	color_rgb(43, 11, 87),	  color_rgb(45, 11, 89),
	color_rgb(47, 10, 91),	  color_rgb(49, 10, 92),
	color_rgb(50, 10, 94),	  color_rgb(52, 10, 95),
	color_rgb(54, 9, 97),	  color_rgb(56, 9, 98),
	color_rgb(57, 9, 99),	  color_rgb(59, 9, 100),
	color_rgb(61, 9, 101),	  color_rgb(62, 9, 102),
	color_rgb(64, 10, 103),	  color_rgb(66, 10, 104),
	color_rgb(68, 10, 104),	  color_rgb(69, 10, 105),
	color_rgb(71, 11, 106),	  color_rgb(73, 11, 106),
	color_rgb(74, 12, 107),	  color_rgb(76, 12, 107),
	color_rgb(77, 13, 108),	  color_rgb(79, 13, 108),
	color_rgb(81, 14, 108),	  color_rgb(82, 14, 109),
	color_rgb(84, 15, 109),	  color_rgb(85, 15, 109),
	color_rgb(87, 16, 110),	  color_rgb(89, 16, 110),
	color_rgb(90, 17, 110),	  color_rgb(92, 18, 110),
	color_rgb(93, 18, 110),	  color_rgb(95, 19, 110),
	color_rgb(97, 19, 110),	  color_rgb(98, 20, 110),
	color_rgb(100, 21, 110),  color_rgb(101, 21, 110),
	color_rgb(103, 22, 110),  color_rgb(105, 22, 110),
	color_rgb(106, 23, 110),  color_rgb(108, 24, 110),
	color_rgb(109, 24, 110),  color_rgb(111, 25, 110),
	color_rgb(113, 25, 110),  color_rgb(114, 26, 110),
	color_rgb(116, 26, 110),  color_rgb(117, 27, 110),
	color_rgb(119, 28, 109),  color_rgb(120, 28, 109),
	color_rgb(122, 29, 109),  color_rgb(124, 29, 109),
	color_rgb(125, 30, 109),  color_rgb(127, 30, 108),
	color_rgb(128, 31, 108),  color_rgb(130, 32, 108),
	color_rgb(132, 32, 107),  color_rgb(133, 33, 107),
	color_rgb(135, 33, 107),  color_rgb(136, 34, 106),
	color_rgb(138, 34, 106),  color_rgb(140, 35, 105),
	color_rgb(141, 35, 105),  color_rgb(143, 36, 105),
	color_rgb(144, 37, 104),  color_rgb(146, 37, 104),
	color_rgb(147, 38, 103),  color_rgb(149, 38, 103),
	color_rgb(151, 39, 102),  color_rgb(152, 39, 102),
	color_rgb(154, 40, 101),  color_rgb(155, 41, 100),
	color_rgb(157, 41, 100),  color_rgb(159, 42, 99),
	color_rgb(160, 42, 99),	  color_rgb(162, 43, 98),
	color_rgb(163, 44, 97),	  color_rgb(165, 44, 96),
	color_rgb(166, 45, 96),	  color_rgb(168, 46, 95),
	color_rgb(169, 46, 94),	  color_rgb(171, 47, 94),
	color_rgb(173, 48, 93),	  color_rgb(174, 48, 92),
	color_rgb(176, 49, 91),	  color_rgb(177, 50, 90),
	color_rgb(179, 50, 90),	  color_rgb(180, 51, 89),
	color_rgb(182, 52, 88),	  color_rgb(183, 53, 87),
	color_rgb(185, 53, 86),	  color_rgb(186, 54, 85),
	color_rgb(188, 55, 84),	  color_rgb(189, 56, 83),
	color_rgb(191, 57, 82),	  color_rgb(192, 58, 81),
	color_rgb(193, 58, 80),	  color_rgb(195, 59, 79),
	color_rgb(196, 60, 78),	  color_rgb(198, 61, 77),
	color_rgb(199, 62, 76),	  color_rgb(200, 63, 75),
	color_rgb(202, 64, 74),	  color_rgb(203, 65, 73),
	color_rgb(204, 66, 72),	  color_rgb(206, 67, 71),
	color_rgb(207, 68, 70),	  color_rgb(208, 69, 69),
	color_rgb(210, 70, 68),	  color_rgb(211, 71, 67),
	color_rgb(212, 72, 66),	  color_rgb(213, 74, 65),
	color_rgb(215, 75, 63),	  color_rgb(216, 76, 62),
	color_rgb(217, 77, 61),	  color_rgb(218, 78, 60),
	color_rgb(219, 80, 59),	  color_rgb(221, 81, 58),
	color_rgb(222, 82, 56),	  color_rgb(223, 83, 55),
	color_rgb(224, 85, 54),	  color_rgb(225, 86, 53),
	color_rgb(226, 87, 52),	  color_rgb(227, 89, 51),
	color_rgb(228, 90, 49),	  color_rgb(229, 92, 48),
	color_rgb(230, 93, 47),	  color_rgb(231, 94, 46),
	color_rgb(232, 96, 45),	  color_rgb(233, 97, 43),
	color_rgb(234, 99, 42),	  color_rgb(235, 100, 41),
	color_rgb(235, 102, 40),  color_rgb(236, 103, 38),
	color_rgb(237, 105, 37),  color_rgb(238, 106, 36),
	color_rgb(239, 108, 35),  color_rgb(239, 110, 33),
	color_rgb(240, 111, 32),  color_rgb(241, 113, 31),
	color_rgb(241, 115, 29),  color_rgb(242, 116, 28),
	color_rgb(243, 118, 27),  color_rgb(243, 120, 25),
	color_rgb(244, 121, 24),  color_rgb(245, 123, 23),
	color_rgb(245, 125, 21),  color_rgb(246, 126, 20),
	color_rgb(246, 128, 19),  color_rgb(247, 130, 18),
	color_rgb(247, 132, 16),  color_rgb(248, 133, 15),
	color_rgb(248, 135, 14),  color_rgb(248, 137, 12),
	color_rgb(249, 139, 11),  color_rgb(249, 140, 10),
	color_rgb(249, 142, 9),	  color_rgb(250, 144, 8),
	color_rgb(250, 146, 7),	  color_rgb(250, 148, 7),
	color_rgb(251, 150, 6),	  color_rgb(251, 151, 6),
	color_rgb(251, 153, 6),	  color_rgb(251, 155, 6),
	color_rgb(251, 157, 7),	  color_rgb(252, 159, 7),
	color_rgb(252, 161, 8),	  color_rgb(252, 163, 9),
	color_rgb(252, 165, 10),  color_rgb(252, 166, 12),
	color_rgb(252, 168, 13),  color_rgb(252, 170, 15),
	color_rgb(252, 172, 17),  color_rgb(252, 174, 18),
	color_rgb(252, 176, 20),  color_rgb(252, 178, 22),
	color_rgb(252, 180, 24),  color_rgb(251, 182, 26),
	color_rgb(251, 184, 29),  color_rgb(251, 186, 31),
	color_rgb(251, 188, 33),  color_rgb(251, 190, 35),
	color_rgb(250, 192, 38),  color_rgb(250, 194, 40),
	color_rgb(250, 196, 42),  color_rgb(250, 198, 45),
	color_rgb(249, 199, 47),  color_rgb(249, 201, 50),
	color_rgb(249, 203, 53),  color_rgb(248, 205, 55),
	color_rgb(248, 207, 58),  color_rgb(247, 209, 61),
	color_rgb(247, 211, 64),  color_rgb(246, 213, 67),
	color_rgb(246, 215, 70),  color_rgb(245, 217, 73),
	color_rgb(245, 219, 76),  color_rgb(244, 221, 79),
	color_rgb(244, 223, 83),  color_rgb(244, 225, 86),
	color_rgb(243, 227, 90),  color_rgb(243, 229, 93),
	color_rgb(242, 230, 97),  color_rgb(242, 232, 101),
	color_rgb(242, 234, 105), color_rgb(241, 236, 109),
	color_rgb(241, 237, 113), color_rgb(241, 239, 117),
	color_rgb(241, 241, 121), color_rgb(242, 242, 125),
	color_rgb(242, 244, 130), color_rgb(243, 245, 134),
	color_rgb(243, 246, 138), color_rgb(244, 248, 142),
	color_rgb(245, 249, 146), color_rgb(246, 250, 150),
	color_rgb(248, 251, 154), color_rgb(249, 252, 157),
	color_rgb(250, 253, 161), color_rgb(252, 255, 164)
};

int inferno_depth_colormap(float relative_depth) {
	relative_depth = std::clamp(relative_depth, 0.0f, 1.0f);
	auto index =
		static_cast<size_t>(relative_depth * (INFERNO_COLOR_COUNT - 1));
	return INFERNO_COLORS[index];
}