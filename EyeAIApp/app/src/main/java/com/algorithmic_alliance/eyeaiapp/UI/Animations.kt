package com.algorithmic_alliance.eyeaiapp.UI

import android.annotation.SuppressLint
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.data.PremiumShapes

@Composable
fun rememberShimmerBrush(backgroundColor: Color, contrastColor: Color): Brush {
    val transition = rememberInfiniteTransition(label = "shimmer")
    val translateAnim by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1000f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "shimmerTranslate"
    )

    val baseColor = lerp(backgroundColor, contrastColor, 0.15f)
    val highlightColor = lerp(backgroundColor, contrastColor, 0.4f)

    val shimmerColors = listOf(baseColor, highlightColor, baseColor)


    return Brush.linearGradient(
        colors = shimmerColors,
        start = Offset(translateAnim - 500f, translateAnim - 500f),
        end = Offset(translateAnim, translateAnim)
    )
}

@SuppressLint("LocalContextGetResourceValueCall")
@Composable
fun ShimmerBox(brush: Brush, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .background(brush)
            .semantics{contentDescription = context.getString(R.string.loading_semantic) }
    )
}

@Composable
fun PremiumButton(
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
    enabled: Boolean = true,
    tonalElevation: Dp = 0.dp,
    shadowElevation: Dp = 3.dp,
    containerColor: Color = MaterialTheme.colorScheme.primary,
    contentColor: Color = MaterialTheme.colorScheme.onPrimary,
    disabledContainerColor: Color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.12f),
    disabledContentColor: Color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f),
    content: @Composable RowScope.() -> Unit
) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.94f else 1f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy),
        label = "buttonScale"
    )

    val buttonSemantic = stringResource(R.string.button_semantic)

    Surface(
        onClick = onClick,
        enabled = enabled,
        shape = PremiumShapes.small,
        color = if (enabled) containerColor else disabledContainerColor,
        contentColor = if (enabled) contentColor else disabledContentColor,
        tonalElevation = if (enabled) tonalElevation else 0.dp,
        shadowElevation = if (enabled) shadowElevation else 0.dp,
        interactionSource = interactionSource,
        modifier = modifier
            .graphicsLayer { scaleX = scale; scaleY = scale }
            .semantics { contentDescription = buttonSemantic },
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 24.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically,
            content = content
        )
    }
}

@Composable
fun PremiumIconButton(
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
    enabled: Boolean = true,
    tonalElevation: Dp = 0.dp,
    shadowElevation: Dp = 0.dp,
    containerColor: Color = Color.Transparent,
    contentColor: Color = MaterialTheme.colorScheme.onSurface,
    disabledContentColor: Color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f),
    content: @Composable () -> Unit
) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.94f else 1f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy),
        label = "buttonScale"
    )

    val buttonSemantic = stringResource(R.string.button_semantic)

    Surface(
        onClick = onClick,
        enabled = enabled,
        shape = PremiumShapes.small,
        color = containerColor,
        contentColor = if (enabled) contentColor else disabledContentColor,
        tonalElevation = if (enabled) tonalElevation else 0.dp,
        shadowElevation = if (enabled) shadowElevation else 0.dp,
        interactionSource = interactionSource,
        modifier = modifier
            .graphicsLayer { scaleX = scale; scaleY = scale }
            .size(40.dp)
            .semantics { contentDescription = buttonSemantic },
    ) {
        Box(contentAlignment = Alignment.Center) {
            content()
        }
    }
}

@Composable
fun PremiumFloatingActionButton(
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
    enabled: Boolean = true,
    tonalElevation: Dp = 0.dp,
    shadowElevation: Dp = 6.dp,
    containerColor: Color = MaterialTheme.colorScheme.primaryContainer,
    contentColor: Color = MaterialTheme.colorScheme.onPrimaryContainer,
    disabledContainerColor: Color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.12f),
    disabledContentColor: Color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f),
    content: @Composable () -> Unit
) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.95f else 1f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy),
        label = "buttonScale"
    )

    val buttonSemantic = stringResource(R.string.button_semantic)

    Surface(
        onClick = onClick,
        enabled = enabled,
        shape = PremiumShapes.small,
        color = if (enabled) containerColor else disabledContainerColor,
        contentColor = if (enabled) contentColor else disabledContentColor,
        tonalElevation = if (enabled) tonalElevation else 0.dp,
        shadowElevation = if (enabled) shadowElevation else 0.dp,
        interactionSource = interactionSource,
        modifier = modifier
            .graphicsLayer { scaleX = scale; scaleY = scale }
            .size(56.dp)
            .semantics { contentDescription = buttonSemantic },
    ) {
        Box(contentAlignment = Alignment.Center) {
            content()
        }
    }
}