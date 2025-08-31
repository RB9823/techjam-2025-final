"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Plus, Trash2, Upload, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";

interface TransitionStep {
  id: string;
  description: string;
  afterScreenshot: string | null;
  isValidating?: boolean;
  validationProgress?: any;
  validationResult?: any;
}

interface UIElement {
  id: string;
  bbox: { x: number; y: number; width: number; height: number };
  caption: string;
  confidence: number;
  detection_method: string;
  clip_similarity?: number;
}

export default function SequenceBuilder() {
  const [initialScreenshot, setInitialScreenshot] = useState<string | null>(
    null
  );
  const [transitions, setTransitions] = useState<TransitionStep[]>([]);

  const handleInitialImageUpload = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const uploadedImage = e.target?.result as string;
        setInitialScreenshot(uploadedImage);

        // Clear existing transitions when new image is uploaded
        setTransitions([]);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAfterImageUpload = (
    stepId: string,
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const afterImageData = e.target?.result as string;
        setTransitions((prev) =>
          prev.map((step) =>
            step.id === stepId
              ? { ...step, afterScreenshot: afterImageData }
              : step
          )
        );

        // Auto-trigger validation for this specific step
        if (initialScreenshot && afterImageData) {
          validateSingleStep(stepId, afterImageData);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const addTransition = () => {
    const newStep: TransitionStep = {
      id: uuidv4(),
      description: "",
      afterScreenshot: null,
    };
    setTransitions((prev) => [...prev, newStep]);
  };

  const removeTransition = (stepId: string) => {
    setTransitions((prev) => prev.filter((step) => step.id !== stepId));
  };

  const updateTransitionDescription = (stepId: string, description: string) => {
    setTransitions((prev) =>
      prev.map((step) => (step.id === stepId ? { ...step, description } : step))
    );
  };

  const [editingId, setEditingId] = useState<string | null>(null);

  const startEditing = (stepId: string) => {
    setEditingId(stepId);
  };

  const stopEditing = () => {
    setEditingId(null);
  };

  const clearInitialImage = () => {
    setInitialScreenshot(null);
  };

  const clearAfterImage = (stepId: string) => {
    setTransitions((prev) =>
      prev.map((step) =>
        step.id === stepId ? { ...step, afterScreenshot: null } : step
      )
    );
  };

  const beforeCanvasRef = useRef<HTMLCanvasElement>(null);
  const afterCanvasRef = useRef<HTMLCanvasElement>(null);

  const drawBoundingBoxes = useCallback(
    (
      canvas: HTMLCanvasElement,
      image: string,
      elements: UIElement[],
      options: { color?: string; stage?: string } = {}
    ) => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const { color = "#3b82f6", stage = "parsing" } = options;

      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        elements.forEach((element) => {
          // Get confidence-based color
          let strokeColor = color;
          if (stage === "filtering") {
            // Purple for CLIP filtering stage
            strokeColor =
              element.clip_similarity && element.clip_similarity > 0.7
                ? "#8b5cf6"
                : "#d1d5db";
          } else if (stage === "analysis") {
            // Orange for change analysis stage
            strokeColor = "#f97316";
          }

          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 3;
          ctx.strokeRect(
            element.bbox.x,
            element.bbox.y,
            element.bbox.width,
            element.bbox.height
          );

          // Draw confidence-based fill
          ctx.fillStyle = strokeColor + "20"; // 20% opacity
          ctx.fillRect(
            element.bbox.x,
            element.bbox.y,
            element.bbox.width,
            element.bbox.height
          );

          // Draw label background
          const labelText = `${element.id} (${(
            element.confidence * 100
          ).toFixed(0)}%)`;
          const metrics = ctx.measureText(labelText);
          const labelWidth = metrics.width + 12;
          const labelHeight = 24;

          ctx.fillStyle = strokeColor;
          ctx.fillRect(
            element.bbox.x,
            element.bbox.y - labelHeight - 4,
            labelWidth,
            labelHeight
          );

          // Draw label text
          ctx.fillStyle = "#fff";
          ctx.font = "12px monospace";
          ctx.fillText(labelText, element.bbox.x + 6, element.bbox.y - 8);
        });
      };
      img.src = image;
    },
    []
  );

  const connectToValidationStream = () => {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket("ws://localhost:8000/api/v1/stream");

      ws.onopen = () => {
        console.log("🚀 Connected to AI validation pipeline");
        resolve(ws);
      };

      ws.onerror = (error) => {
        console.error("❌ WebSocket connection failed:", error);
        reject(error);
      };

      return ws;
    });
  };

  const validateSingleStep = async (stepId: string, afterImageData: string) => {
    // Mark this step as validating
    setTransitions((prev) =>
      prev.map((step) =>
        step.id === stepId
          ? {
              ...step,
              isValidating: true,
              validationProgress: null,
              validationResult: null,
            }
          : step
      )
    );

    try {
      const ws = (await connectToValidationStream()) as WebSocket;

      const step = transitions.find((t) => t.id === stepId);
      if (!step) return;

      // Set up message handler for this specific step
      const messageHandler = (event: MessageEvent) => {
        const message = JSON.parse(event.data);

        if (message.type === "progress") {
          setTransitions((prev) =>
            prev.map((t) =>
              t.id === stepId ? { ...t, validationProgress: message.data } : t
            )
          );
        } else if (message.type === "result") {
          setTransitions((prev) =>
            prev.map((t) =>
              t.id === stepId
                ? { ...t, isValidating: false, validationResult: message.data }
                : t
            )
          );
          ws.removeEventListener("message", messageHandler);
          ws.close();
        } else if (message.type === "error") {
          console.error(`Validation error for step ${stepId}:`, message.data);
          setTransitions((prev) =>
            prev.map((t) =>
              t.id === stepId ? { ...t, isValidating: false } : t
            )
          );
          ws.removeEventListener("message", messageHandler);
          ws.close();
        }
      };

      ws.addEventListener("message", messageHandler);

      // Send validation request for this step
      const request = {
        type: "validate",
        data: {
          qa_prompt: step.description || "Validate this UI transition",
          before_image_base64: initialScreenshot || "",
          after_image_base64: afterImageData,
        },
      };

      ws.send(JSON.stringify(request));
    } catch (error) {
      console.error(`Failed to validate step ${stepId}:`, error);
      setTransitions((prev) =>
        prev.map((step) =>
          step.id === stepId ? { ...step, isValidating: false } : step
        )
      );
    }
  };

  // Draw Canvas function that can be called reactively
  const drawStepCanvas = useCallback(
    (step: TransitionStep, stepIndex: number) => {
      console.log(`🎯 Drawing Canvas for step ${stepIndex + 1} (${step.id})`);

      // Draw before Canvas
      const beforeCanvas = document.getElementById(
        `beforeCanvas_${step.id}`
      ) as HTMLCanvasElement;
      if (
        beforeCanvas &&
        initialScreenshot &&
        (step.isValidating || step.validationResult)
      ) {
        console.log(`🖼️ Drawing before Canvas for step ${stepIndex + 1}`);

        // Small delay to ensure Canvas is rendered
        setTimeout(() => {
          const ctx = beforeCanvas.getContext("2d");
          if (ctx) {
            const img = new Image();
            img.onload = () => {
              beforeCanvas.width = img.width;
              beforeCanvas.height = img.height;
              ctx.drawImage(img, 0, 0);

              // Draw bounding boxes based on current validation stage
              if (step.validationProgress) {
                const stage = step.validationProgress.stage;
                let elements = [];
                let color = "#3b82f6";

                console.log(
                  `🎨 Stage: ${stage}, Full progress object:`,
                  JSON.stringify(step.validationProgress, null, 2)
                );

                // Access elements data directly from validation progress
                const beforeElements =
                  step.validationProgress.before_elements || [];
                const filteredElements =
                  step.validationProgress.filtered_elements || [];
                const afterElements =
                  step.validationProgress.after_elements || [];
                const filteredAfterElements =
                  step.validationProgress.filtered_after_elements || [];

                console.log(`🔍 Elements found:`, {
                  beforeElements: beforeElements.length,
                  filteredElements: filteredElements.length,
                  afterElements: afterElements.length,
                  filteredAfterElements: filteredAfterElements.length,
                  stage,
                });

                if (stage === "parsing" && beforeElements.length > 0) {
                  elements = beforeElements;
                  color = "#3b82f6";
                  console.log("📦 Using parsing elements:", elements.length);
                } else if (
                  stage === "filtering" &&
                  filteredElements.length > 0
                ) {
                  elements = filteredElements;
                  color = "#8b5cf6";
                  console.log("🔽 Using filtered elements:", elements.length);
                } else if (stage === "analysis" && beforeElements.length > 0) {
                  elements = beforeElements;
                  color = "#f97316";
                  console.log(
                    "🔍 Using before elements for analysis:",
                    elements.length
                  );
                } else if (beforeElements.length > 0) {
                  // Fallback: use before elements for any stage
                  elements = beforeElements;
                  color = "#3b82f6";
                  console.log(
                    "🔧 Using fallback before elements:",
                    elements.length
                  );
                }

                console.log(
                  `🎯 Drawing ${elements.length} bounding boxes with color ${color}`
                );

                // Draw bounding boxes
                elements.forEach((element: UIElement, i: number) => {
                  console.log(
                    `📍 Drawing element ${i + 1}:`,
                    element.id,
                    element.bbox
                  );

                  ctx.strokeStyle = color;
                  ctx.lineWidth = 3;
                  ctx.strokeRect(
                    element.bbox.x,
                    element.bbox.y,
                    element.bbox.width,
                    element.bbox.height
                  );

                  ctx.fillStyle = color + "30";
                  ctx.fillRect(
                    element.bbox.x,
                    element.bbox.y,
                    element.bbox.width,
                    element.bbox.height
                  );

                  const labelText = `${element.id} (${(
                    element.confidence * 100
                  ).toFixed(0)}%)`;
                  ctx.fillStyle = color;
                  ctx.fillRect(
                    element.bbox.x,
                    element.bbox.y - 20,
                    labelText.length * 8,
                    18
                  );
                  ctx.fillStyle = "#fff";
                  ctx.font = "11px monospace";
                  ctx.fillText(
                    labelText,
                    element.bbox.x + 4,
                    element.bbox.y - 6
                  );
                });
              }
            };
            img.src = initialScreenshot;
          }
        }, 100); // Small delay to ensure DOM is ready
      }

      // Draw after Canvas
      const afterCanvas = document.getElementById(
        `afterCanvas_${step.id}`
      ) as HTMLCanvasElement;
      if (
        afterCanvas &&
        step.afterScreenshot &&
        (step.isValidating || step.validationResult)
      ) {
        console.log(`🖼️ Drawing after Canvas for step ${stepIndex + 1}`);

        setTimeout(() => {
          const ctx = afterCanvas.getContext("2d");
          if (ctx) {
            const img = new Image();
            img.onload = () => {
              afterCanvas.width = img.width;
              afterCanvas.height = img.height;
              ctx.drawImage(img, 0, 0);

              // Draw after elements - either from intermediate progress or final result
              let afterElementsToRender = [];
              let afterColor = "#10b981"; // Default green for final result

              if (step.validationResult?.after_elements) {
                // Final validation complete - use final after elements
                afterElementsToRender = step.validationResult.after_elements;
                afterColor = "#10b981";
                console.log(
                  `🎯 Drawing ${afterElementsToRender.length} final after elements`
                );
              } else if (step.validationProgress) {
                // Intermediate stage - use appropriate elements for after image
                const stage = step.validationProgress.stage;
                if (
                  stage === "parsing" &&
                  step.validationProgress.after_elements
                ) {
                  afterElementsToRender =
                    step.validationProgress.after_elements;
                  afterColor = "#3b82f6";
                  console.log(
                    `🎯 Drawing ${afterElementsToRender.length} parsing after elements`
                  );
                } else if (
                  stage === "filtering" &&
                  step.validationProgress.filtered_after_elements
                ) {
                  afterElementsToRender =
                    step.validationProgress.filtered_after_elements;
                  afterColor = "#8b5cf6";
                  console.log(
                    `🎯 Drawing ${afterElementsToRender.length} filtered after elements`
                  );
                } else if (
                  stage === "analysis" &&
                  step.validationProgress.after_elements
                ) {
                  afterElementsToRender =
                    step.validationProgress.after_elements;
                  afterColor = "#f97316";
                  console.log(
                    `🎯 Drawing ${afterElementsToRender.length} analysis after elements`
                  );
                }
              }

              if (afterElementsToRender.length > 0) {
                afterElementsToRender.forEach(
                  (element: UIElement, i: number) => {
                    console.log(
                      `📍 After element ${i + 1}:`,
                      element.id,
                      element.bbox
                    );

                    ctx.strokeStyle = afterColor;
                    ctx.lineWidth = 3;
                    ctx.strokeRect(
                      element.bbox.x,
                      element.bbox.y,
                      element.bbox.width,
                      element.bbox.height
                    );

                    ctx.fillStyle = afterColor + "30";
                    ctx.fillRect(
                      element.bbox.x,
                      element.bbox.y,
                      element.bbox.width,
                      element.bbox.height
                    );

                    const labelText = `${element.id} (${(
                      element.confidence * 100
                    ).toFixed(0)}%)`;
                    ctx.fillStyle = afterColor;
                    ctx.fillRect(
                      element.bbox.x,
                      element.bbox.y - 20,
                      labelText.length * 8,
                      18
                    );
                    ctx.fillStyle = "#fff";
                    ctx.font = "11px monospace";
                    ctx.fillText(
                      labelText,
                      element.bbox.x + 4,
                      element.bbox.y - 6
                    );
                  }
                );
              }
            };
            img.src = step.afterScreenshot || "";
          }
        }, 100);
      }
    },
    [initialScreenshot]
  );

  // Progressive bbox visualization - trigger on any step state change
  useEffect(() => {
    console.log("🎯 Transitions changed - updating all Canvas visualizations");
    transitions.forEach((step, stepIndex) => {
      if (step.isValidating || step.validationResult) {
        drawStepCanvas(step, stepIndex);
      }
    });
  }, [transitions, drawStepCanvas]);

  // Specific trigger for validation progress updates (to catch nested state changes)
  useEffect(() => {
    const hasValidatingSteps = transitions.some((step) => step.isValidating);
    const progressStages = transitions
      .map((step) => step.validationProgress?.stage)
      .filter(Boolean);

    console.log("🔄 Validation progress changed:", {
      hasValidatingSteps,
      progressStages,
    });

    if (hasValidatingSteps) {
      transitions.forEach((step, stepIndex) => {
        if (step.validationProgress) {
          console.log(
            `🎨 Re-drawing Canvas for step ${
              stepIndex + 1
            } due to progress change:`,
            step.validationProgress.stage
          );
          drawStepCanvas(step, stepIndex);
        }
      });
    }
  }, [
    transitions.map((step) => step.validationProgress?.stage),
    transitions.map((step) => step.validationProgress?.before_elements?.length),
    transitions.map(
      (step) => step.validationProgress?.filtered_elements?.length
    ),
    drawStepCanvas,
  ]);

  // Initialize Canvas images when validation starts (show base images immediately)
  useEffect(() => {
    transitions.forEach((step) => {
      if (step.isValidating && !step.validationProgress) {
        // Show base images immediately when validation starts
        const beforeCanvas = document.getElementById(
          `beforeCanvas_${step.id}`
        ) as HTMLCanvasElement;
        if (beforeCanvas && initialScreenshot) {
          const ctx = beforeCanvas.getContext("2d");
          if (ctx) {
            const img = new Image();
            img.onload = () => {
              beforeCanvas.width = img.width;
              beforeCanvas.height = img.height;
              ctx.drawImage(img, 0, 0);
            };
            img.src = initialScreenshot;
          }
        }

        const afterCanvas = document.getElementById(
          `afterCanvas_${step.id}`
        ) as HTMLCanvasElement;
        if (afterCanvas && step.afterScreenshot) {
          const ctx = afterCanvas.getContext("2d");
          if (ctx) {
            const img = new Image();
            img.onload = () => {
              afterCanvas.width = img.width;
              afterCanvas.height = img.height;
              ctx.drawImage(img, 0, 0);
            };
            img.src = step.afterScreenshot;
          }
        }
      }
    });
  }, [transitions, initialScreenshot]);

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-3">
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold mb-2">
          AI-Powered UI Analysis & Sequence Builder
        </h1>
        <p className="text-muted-foreground">
          Upload any UI screenshot for instant AI element detection, then build
          sequences to analyze UI changes
        </p>
      </div>

      {/* Flow Container */}
      <div className="space-y-3">
        {/* Initial Image Card */}
        <Card className="border-2 border-dashed border-gray-200 py-0">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <Label className="text-lg font-semibold">
                  Upload UI Screenshot
                </Label>
                <p className="text-sm text-muted-foreground mt-1">
                  🚀 AI analysis begins immediately after upload
                </p>
              </div>
              <div className="flex items-center gap-3">
                {initialScreenshot ? (
                  <div className="relative">
                    <div className="w-20 h-20 rounded-lg border overflow-hidden">
                      <img
                        src={initialScreenshot}
                        alt="Initial UI state"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={clearInitialImage}
                      className="absolute -top-2 -right-2 h-6 w-6 p-0 bg-white border border-gray-200 rounded-full hover:bg-gray-50 cursor-pointer"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                ) : (
                  <Label
                    htmlFor="initial-screenshot"
                    className="cursor-pointer"
                  >
                    <div className="w-20 h-20 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center hover:border-gray-400 transition-colors">
                      <Upload className="h-5 w-5 text-gray-400" />
                    </div>
                  </Label>
                )}
                <Input
                  id="initial-screenshot"
                  type="file"
                  accept="image/*"
                  onChange={handleInitialImageUpload}
                  className="hidden"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Transition Steps */}
        {transitions.map((step, index) => (
          <div key={step.id} className="space-y-2">
            {/* Connecting Line */}
            <div className="flex justify-center">
              <div className="w-0.5 h-6 bg-gray-200"></div>
            </div>

            {/* Transition Card */}
            <Card
              className={`border-2 py-0 transition-all duration-300 ${
                step.isValidating
                  ? "border-blue-400 bg-blue-50"
                  : step.validationResult
                  ? step.validationResult.is_valid
                    ? "border-green-400 bg-green-50"
                    : "border-red-400 bg-red-50"
                  : "border-dashed border-gray-200"
              }`}
            >
              <CardContent className="p-4">
                <div className="space-y-4">
                  {/* Step Header */}
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          {editingId === step.id ? (
                            <Textarea
                              placeholder="Describe the action to be performed"
                              value={step.description}
                              onChange={(e) =>
                                updateTransitionDescription(
                                  step.id,
                                  e.target.value
                                )
                              }
                              onBlur={stopEditing}
                              onKeyDown={(e) => {
                                if (e.key === "Enter" && !e.shiftKey) {
                                  e.preventDefault();
                                  stopEditing();
                                }
                                if (e.key === "Escape") {
                                  stopEditing();
                                }
                              }}
                              className="min-h-[50px] resize-none text-lg font-semibold"
                              autoFocus
                            />
                          ) : (
                            <div
                              className="text-lg font-semibold px-2 py-1 rounded transition-colors min-h-[28px] flex items-center cursor-pointer hover:bg-gray-50"
                              onClick={() => startEditing(step.id)}
                            >
                              {step.description ||
                                "Click to add description..."}
                            </div>
                          )}
                          <div className="flex gap-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => removeTransition(step.id)}
                              className="h-6 w-6 p-0 text-red-600 hover:text-red-700 cursor-pointer"
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 ml-4">
                      {step.afterScreenshot ? (
                        <div className="relative">
                          <div className="w-20 h-20 rounded-lg border overflow-hidden">
                            <img
                              src={step.afterScreenshot}
                              alt={`After state for step ${index + 1}`}
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => clearAfterImage(step.id)}
                            className="absolute -top-2 -right-2 h-6 w-6 p-0 bg-white border border-gray-200 rounded-full hover:bg-gray-50 cursor-pointer"
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      ) : (
                        <Label
                          htmlFor={`after-screenshot-${step.id}`}
                          className="cursor-pointer"
                        >
                          <div className="w-20 h-20 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center hover:border-gray-400 transition-colors">
                            <Upload className="h-5 w-5 text-gray-400" />
                          </div>
                        </Label>
                      )}
                      <Input
                        id={`after-screenshot-${step.id}`}
                        type="file"
                        accept="image/*"
                        onChange={(e) => handleAfterImageUpload(step.id, e)}
                        className="hidden"
                      />
                    </div>
                  </div>

                  {/* Step Validation Progress */}
                  {step.isValidating && step.validationProgress && (
                    <div className="space-y-3 bg-blue-100 p-3 rounded-lg">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                        <span className="text-sm font-medium text-blue-800">
                          {step.validationProgress.stage?.toUpperCase()} -{" "}
                          {step.validationProgress.message}
                        </span>
                      </div>
                      <div className="w-full bg-blue-200 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                          style={{
                            width: `${
                              step.validationProgress.progress_percent || 0
                            }%`,
                          }}
                        ></div>
                      </div>
                      {step.validationProgress.elements_detected && (
                        <div className="text-xs text-blue-700">
                          🔍 {step.validationProgress.elements_detected}{" "}
                          elements detected
                        </div>
                      )}
                    </div>
                  )}

                  {/* Step Validation Result */}
                  {step.validationResult && (
                    <div
                      className={`space-y-3 p-3 rounded-lg ${
                        step.validationResult.is_valid
                          ? "bg-green-100"
                          : "bg-red-100"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span
                          className={`text-sm font-medium ${
                            step.validationResult.is_valid
                              ? "text-green-800"
                              : "text-red-800"
                          }`}
                        >
                          {step.validationResult.is_valid
                            ? "✅ Validation Complete"
                            : "❌ Validation Failed"}
                        </span>
                        <span
                          className={`px-2 py-1 rounded text-xs ${
                            step.validationResult.is_valid
                              ? "bg-green-200 text-green-800"
                              : "bg-red-200 text-red-800"
                          }`}
                        >
                          {step.validationResult.is_valid ? "PASSED" : "FAILED"}
                          ({(step.validationResult.confidence * 100).toFixed(0)}
                          %)
                        </span>
                      </div>
                      {/* <div className="text-xs text-green-700">
                        {step.validationResult.detected_changes?.length || 0}{" "}
                        errors found in{" "}
                        {step.validationResult.processing_time_seconds?.toFixed(
                          1
                        )}
                        s
                      </div> */}
                    </div>
                  )}

                  {/* Individual Step Canvas Visualization */}
                  {(step.isValidating || step.validationResult) &&
                    step.afterScreenshot && (
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <h5 className="text-xs font-medium text-gray-600 mb-1">
                            Before Analysis
                            {step.validationProgress?.stage && (
                              <span className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-800 rounded text-xs">
                                {step.validationProgress.stage.toUpperCase()}
                              </span>
                            )}
                          </h5>
                          <div className="relative bg-gray-900 rounded p-2">
                            <canvas
                              id={`beforeCanvas_${step.id}`}
                              width="200"
                              height="150"
                              className="w-full border rounded bg-black"
                            />
                            {step.validationProgress?.before_elements && (
                              <div className="absolute bottom-1 left-1 text-xs text-white bg-black bg-opacity-50 px-1 py-0.5 rounded">
                                {step.validationProgress.stage ===
                                  "filtering" &&
                                step.validationProgress.filtered_elements
                                  ? step.validationProgress.filtered_elements
                                      .length
                                  : step.validationProgress.before_elements
                                      .length}{" "}
                                elements
                              </div>
                            )}
                          </div>
                        </div>
                        <div>
                          <h5 className="text-xs font-medium text-gray-600 mb-1">
                            After Analysis
                            {step.validationResult && (
                              <span
                                className={`ml-2 px-2 py-0.5 rounded text-xs ${
                                  step.validationResult.is_valid
                                    ? "bg-green-100 text-green-800"
                                    : "bg-red-100 text-red-800"
                                }`}
                              >
                                {step.validationResult.is_valid
                                  ? "PASSED"
                                  : "FAILED"}
                              </span>
                            )}
                          </h5>
                          <div className="relative bg-gray-900 rounded p-2">
                            <canvas
                              id={`afterCanvas_${step.id}`}
                              width="200"
                              height="150"
                              className="w-full border rounded bg-black"
                            />
                            {step.validationResult?.after_elements && (
                              <div className="absolute bottom-1 left-1 text-xs text-white bg-black bg-opacity-50 px-1 py-0.5 rounded">
                                {step.validationResult.after_elements.length}{" "}
                                elements
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                </div>
              </CardContent>
            </Card>
          </div>
        ))}

        {/* Add New Transition Card */}
        <div className="space-y-2">
          {transitions.length > 0 && (
            <div className="flex justify-center">
              <div className="w-0.5 h-6 bg-gray-200"></div>
            </div>
          )}
          <Card
            className="border-2 border-dashed border-gray-200 hover:border-gray-300 transition-colors cursor-pointer py-0"
            onClick={addTransition}
          >
            <CardContent className="p-4">
              <div className="w-full h-full min-h-[60px] flex flex-col items-center justify-center gap-2 text-gray-500 hover:text-gray-700">
                <Plus className="h-6 w-6" />
                <span className="text-lg font-semibold">
                  Add Transition Step
                </span>
                <span className="text-xs text-center">
                  Compare UI changes between states
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Auto-Validation Status */}
      <div className="flex justify-center pt-4">
        <Card className="bg-gradient-to-r from-blue-50 to-purple-50">
          <CardContent className="p-4 text-center">
            <div className="text-sm text-muted-foreground">
              🤖 AI analysis starts immediately when you upload the initial
              image
            </div>
            {!initialScreenshot ? (
              <div className="text-xs text-orange-600 mt-1">
                ⬆️ Upload an image to see instant UI element detection and
                analysis!
              </div>
            ) : (
              <div className="text-xs text-green-600 mt-1">
                ✨ Add more transition steps to analyze UI changes between
                states
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
