import { useState, useEffect } from 'react'
import { supabase, getUser } from '../lib/supabase'
import type { Page } from '../App'

interface SkillEditorProps {
  skillName?: string
  onNavigate: (page: Page) => void
}

const DEFAULT_CONTENT = `## Trigger
- [Describe when this procedure applies]

## Prerequisites
- [Required tools, permissions, or data]

## Logical Steps
1. **[Step name]**: [Instruction]
   - Validation: [How to verify this step worked]

2. **[Step name]**: [Instruction]
   - Expected outcome: [What success looks like]

## Constraints
- [Rules the agent must follow]
- **Hard limit:** Never [action] without [condition]

## References
- [Links to related skills or documentation]
`

export default function SkillEditor({ skillName, onNavigate }: SkillEditorProps) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [content, setContent] = useState(DEFAULT_CONTENT)
  const [status, setStatus] = useState('draft')
  const [tags, setTags] = useState('')
  const [requiresApproval, setRequiresApproval] = useState(false)
  const [preview, setPreview] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const isEditing = !!skillName

  useEffect(() => {
    if (skillName) {
      async function load() {
        const { data } = await supabase
          .from('skills')
          .select('*')
          .eq('name', skillName)
          .single()

        if (data) {
          setName(data.name)
          setDescription(data.description)
          setContent(data.content)
          setStatus(data.status)
          setTags(data.tags?.join(', ') || '')
          setRequiresApproval(data.requires_approval || false)
        }
      }
      load()
    }
  }, [skillName])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setSaving(true)

    const user = await getUser()
    const tagList = tags.split(',').map(t => t.trim()).filter(Boolean)

    const payload = {
      name: name.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, ''),
      description,
      content,
      status,
      author: user?.email || 'unknown',
      tags: tagList,
      requires_approval: requiresApproval,
      last_validated: new Date().toISOString(),
      ...(status === 'active' && !isEditing ? { approved_by: user?.email || 'unknown', approved_at: new Date().toISOString() } : {})
    }

    if (isEditing) {
      const { error: err } = await supabase
        .from('skills')
        .update(payload)
        .eq('name', skillName)

      if (err) setError(err.message)
      else onNavigate({ name: 'skill-view', skillName: payload.name })
    } else {
      const { error: err } = await supabase
        .from('skills')
        .insert(payload)

      if (err) setError(err.message)
      else onNavigate({ name: 'skill-view', skillName: payload.name })
    }

    setSaving(false)
  }

  return (
    <div style={{ padding: '2rem', maxWidth: '800px' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '2rem'
      }}>
        <h1>{isEditing ? 'Edit Skill' : 'New Skill'}</h1>
        <button
          className="btn btn-ghost btn-sm"
          onClick={() => onNavigate(isEditing ? { name: 'skill-view', skillName: skillName! } : { name: 'skills' })}
        >
          Cancel
        </button>
      </div>

      {error && (
        <div style={{
          padding: '0.75rem',
          background: 'rgba(239, 68, 68, 0.08)',
          border: '1px solid rgba(239, 68, 68, 0.2)',
          color: '#EF4444',
          fontSize: '0.8125rem',
          marginBottom: '1.5rem',
          fontFamily: 'var(--font-mono)'
        }}>
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        {/* Name */}
        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{
            display: 'block',
            fontSize: '0.6875rem',
            fontWeight: 600,
            color: 'var(--text-tertiary)',
            marginBottom: '0.375rem',
            textTransform: 'uppercase',
            letterSpacing: '0.04em'
          }}>
            Name <span style={{ color: 'var(--text-muted)' }}>(kebab-case)</span>
          </label>
          <input
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            required
            disabled={isEditing}
            placeholder="deploy-api-to-staging"
            style={{ fontFamily: 'var(--font-mono)' }}
          />
        </div>

        {/* Description */}
        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{
            display: 'block',
            fontSize: '0.6875rem',
            fontWeight: 600,
            color: 'var(--text-tertiary)',
            marginBottom: '0.375rem',
            textTransform: 'uppercase',
            letterSpacing: '0.04em'
          }}>
            Description
          </label>
          <input
            type="text"
            value={description}
            onChange={e => setDescription(e.target.value)}
            required
            maxLength={200}
            placeholder="How to deploy the API service to staging"
          />
        </div>

        {/* Tags + Status Row */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '2fr 1fr 1fr',
          gap: '1rem',
          marginBottom: '1.25rem'
        }}>
          <div>
            <label style={{
              display: 'block',
              fontSize: '0.6875rem',
              fontWeight: 600,
              color: 'var(--text-tertiary)',
              marginBottom: '0.375rem',
              textTransform: 'uppercase',
              letterSpacing: '0.04em'
            }}>
              Tags
            </label>
            <input
              type="text"
              value={tags}
              onChange={e => setTags(e.target.value)}
              placeholder="devops, deployment, api"
            />
          </div>

          <div>
            <label style={{
              display: 'block',
              fontSize: '0.6875rem',
              fontWeight: 600,
              color: 'var(--text-tertiary)',
              marginBottom: '0.375rem',
              textTransform: 'uppercase',
              letterSpacing: '0.04em'
            }}>
              Status
            </label>
            <select
              value={status}
              onChange={e => setStatus(e.target.value)}
              style={{ width: '100%' }}
            >
              <option value="draft">Draft</option>
              <option value="active">Active</option>
              <option value="deprecated">Deprecated</option>
              <option value="pending_review">Pending Review</option>
            </select>
          </div>

          <div>
            <label style={{
              display: 'block',
              fontSize: '0.6875rem',
              fontWeight: 600,
              color: 'var(--text-tertiary)',
              marginBottom: '0.375rem',
              textTransform: 'uppercase',
              letterSpacing: '0.04em'
            }}>
              Approval
            </label>
            <label style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 0',
              cursor: 'pointer',
              fontSize: '0.8125rem',
              color: 'var(--text-secondary)'
            }}>
              <input
                type="checkbox"
                checked={requiresApproval}
                onChange={e => setRequiresApproval(e.target.checked)}
              />
              Requires approval
            </label>
          </div>
        </div>

        {/* Content */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '0.375rem'
          }}>
            <label style={{
              fontSize: '0.6875rem',
              fontWeight: 600,
              color: 'var(--text-tertiary)',
              textTransform: 'uppercase',
              letterSpacing: '0.04em'
            }}>
              Content <span style={{ color: 'var(--text-muted)' }}>(Markdown)</span>
            </label>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => setPreview(!preview)}
            >
              {preview ? 'Edit' : 'Preview'}
            </button>
          </div>

          {preview ? (
            <div
              className="markdown-content card"
              style={{ padding: '1.25rem', minHeight: '300px' }}
              dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }}
            />
          ) : (
            <textarea
              value={content}
              onChange={e => setContent(e.target.value)}
              required
              style={{
                minHeight: '400px',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.8125rem',
                lineHeight: 1.6
              }}
            />
          )}
        </div>

        {/* Actions */}
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button
            type="submit"
            disabled={saving}
            className="btn btn-primary"
          >
            {saving ? 'Saving...' : isEditing ? 'Update Skill' : 'Create Skill'}
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => onNavigate(isEditing ? { name: 'skill-view', skillName: skillName! } : { name: 'skills' })}
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  )
}

// Simple markdown renderer for preview
function renderMarkdown(md: string): string {
  return md
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^\*\*(.*)\*\*/gim, '<strong>$1</strong>')
    .replace(/^- (.*$)/gim, '<li>$1</li>')
    .replace(/`([^`]+)`/gim, '<code>$1</code>')
    .replace(/\n/gim, '<br>')
}
