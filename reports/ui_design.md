# Road Damage Detection UI Design Brief

## Direction

A compact dark-mode field-inspection console for civil engineers, road auditors, and project reviewers. The visual identity is utilitarian infrastructure: matte asphalt black, lane-marking amber, inspection green, and restrained steel accents. The interface should feel operational, durable, and trustworthy rather than decorative.

## Primary Journey

1. Load the selected production model status.
2. Upload or capture a road image.
3. Adjust confidence, overlap, class filters, and detection limit.
4. Run detection.
5. Review annotated image, detection counts, timing, and raw rows.
6. Download annotated image or detection data.

## Layout

The first viewport prioritizes production status, the current model runtime, image input, controls, and the run action. Results appear directly below in a two-column comparison layout on desktop and a single column on mobile.

## Tokens

- Background: `#0b0d0c`
- Surface: `#111512`
- Raised surface: `#171c18`
- Ink: `#eef3ee`
- Muted text: `#a8b2aa`
- Asphalt charcoal: `#0f1210`
- Lane amber: `#f1c453`
- Inspection green: `#45d483`
- Warning: `#ff6b5f`
- Border: `#2f3832`
- Radius: `8px`
- Spacing unit: `8px`

## Components And States

Controls are grouped in the sidebar. Status uses distinct tones: amber for attention, green for healthy local runtime, red for warnings or errors, and neutral dark surfaces for model metadata. Detection metrics use restrained borders and class colors for scanability.

## Accessibility

Text must keep strong contrast against panel backgrounds. Controls retain native keyboard behavior. Important state changes are shown with text and color, not color alone. The app avoids dense copy and keeps controls close to the image workflow.

## Responsive Behavior

Desktop uses two result columns for original and annotated images. Narrow screens collapse into a single column with full-width controls and download buttons.
